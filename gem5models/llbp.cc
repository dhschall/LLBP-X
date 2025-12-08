/*
 * Copyright (c) 2024 The University of Edinburgh
 * Copyright (c) 2025 Technical University of Munich
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: David Schall
 *
 * Implementation of the Last-level Branch Predictor (LLBP)
 * https://dhschall.github.io/assets/pdf/LLBP_MICRO24.pdf
 *
 * Includes also the enhanced features of LLBP-X the dynamic context depth
 * adaption which is enabled by the `adaptCtxDepth` flag.
 *
 *
 */

#include "cpu/pred/llbp.hh"
#include "debug/LLBP.hh"
#include <algorithm>
#include <sstream>


namespace gem5
{

namespace branch_prediction
{

LLBP::LLBP(const LLBPParams &params)
  : ConditionalPredictor(params),
    Clocked(*params.clk_domain),
    base(params.base),
    llbpStorage(params.backingStorageCapacity, params.patternSetCapacity,
                params.backingStorageAssoc, params.patternSetAssoc),
    patternBuffer(params.patternBufferCapacity, params.patternBufferAssoc),
    backingStorageLatency(params.backingStorageLatency),
    patternSetCapacity(params.patternSetCapacity),
    patternSetAssoc(params.patternSetAssoc),
    contextCounterWidth(params.contextCounterWidth),
    patternCounterWidth(params.patternCounterWidth),
    TTWidth(params.patterTagBits),
    optimalPrefetching(params.optimalPrefetching),
    unlimited(params.unlimited),
    ctt(params.citCapacity, params.citAssoc),
    adaptCtxDepth(params.adaptCtxDepth),
    trackingThreshold(7),
    adaptThreshold(7),
    histLenThreshold(22),
    rcr(params.rcrType,
        params.rcrWindow,
        params.rcrDist,
        params.rcrShift,
        params.rcrTagWidth,
        params.rcrBaseTagWidth),
    stats(this)
{
    // assert(floorLog2(patternSetAssoc)
    // TODO: Add assert to check that the base predictor is of type LLBP_TAGE_64KB
    static_cast<LLBP_TAGE_64KB *>(base->tage)->setParent(this);
    DPRINTF(LLBP, "Using LLBP\n");
    DPRINTF(LLBP, "RCR: T=%d, W=%d, D=%d, S=%d, tagWidthBits=%d, tagBaseWidthBits=%d\n",
            rcr.T, rcr.W, rcr.D, rcr.S, params.rcrTagWidth, params.rcrBaseTagWidth);
    // DPRINTF(LLBP, "RCR: context hash config: [T:%i, W:%i, D:%i, S:%i, CTWidth:%i]\n",
    //         T, W, D, S, CTWidth);
    DPRINTF(LLBP, "Storage: cap=%d, bits=%d\n",
            params.backingStorageCapacity, contextCounterWidth);

    printf("Pattern Store:\n");
    llbpStorage.printCfg();
    printf("Pattern Buffer:\n");
    patternBuffer.printCfg();
}

void
LLBP::init()
{
    // First initialize the base predictor
    base->tage->init();
    speculativeHistUpdate = base->tage->isSpeculativeUpdateEnabled();


    for (int i = 0; i < nW; i++)
        fltTables[i].resize(base->getNumHistoryTables() + 1, -1);

    if (!adaptCtxDepth) {
        // LLBP History length selection ----------------------------
        //
        // LLBP does not provide for all different history lenghts in
        // TAGE a prediction only for the following once which where
        // empirically determined. Note this
        // are not the actual length but the table indices in TAGE.
        auto l = {6,10,13,14,15,16,17,18,  19,20,22,24,26,28,32,36};
        // auto l = {2,6,10,13,14,15,17,18,  19,20,22,24,26,28,32,36};

        int n = 0;
        for (auto i : l) {
            // To reduce the complexity of the multiplexer LLBP groups
            // always four consecutive history lenght in one bucket.
            // As the pattern sets are implemented a set associative
            // structure the lower bits determine the set=bucket.
            // The `fltTable`-map not only filters the history lengths
            // but also maps each length the correct pattern set index.
            // E.e. for the four way associativity the following function
            // ensures that history length 6,10,13,14 gets assign
            // 0,4,8,12 with the lowest two bits 0b00. Thus, the set will
            // be the same.
            auto pa = (patternSetAssoc == patternSetCapacity) ?
                    1 : patternSetAssoc;
            auto bucket = n / pa;
            fltTables[0][i] = ((i) << ceilLog2(pa) ) | bucket;
            printf("%i=>%i:%i:%i ", i, n, bucket, fltTables[0][i]);
            n++;
        }
    } else {

        // LLBP History length selection ----------------------------
        // With adaption we use two different history ranges.
        // For shallow context only allocate histories up to history length 22
        // and for deep contexts drop all histories shorter than 12.
        int n = 0;
        for (int i = 1; i <= base->getNumHistoryTables(); i++) {
            if (base->tage->noSkip[i]) {

                if (i > 22) break;

                auto dnom = 4;

                auto bucket = (n / dnom) % 4;
                fltTables[0][i] = ((n) << 2) | bucket;
                printf("%i=>%i:%i:%i ", i, n, bucket, fltTables[0][i]);
                n++;
            }
        }

        for (int i = 12; i <= base->getNumHistoryTables(); i++) {
            if (base->tage->noSkip[i]) {

                auto dnom = 4;

                auto bucket = (n / dnom) % 4;
                fltTables[nW-1][i] = ((n) << 2) | bucket;
                printf("%i=>%i:%i:%i ", i, n, bucket, fltTables[nW-1][i]);
                n++;
            }
        }
    }

    printf("\n");
}


bool
LLBP::lookup(ThreadID tid, Addr pc, void *&bp_history)
{


    LLBPBranchInfo *bi = new LLBPBranchInfo(pc, true);
    bp_history = (void*)(bi);

    bi->base_pred = base->predict(tid, pc, true, bi->ltage_bi);

    auto scltage_bi = static_cast<TAGE_SC_L::TageSCLBranchInfo*>(bi->ltage_bi);
    auto tage_bi = scltage_bi->tageBranchInfo;


    llbpPredict(tid, pc, bi);


    // Arbitrate between TAGE and LLBP
    if (bi->hit && (bi->hitIndex >= tage_bi->hitBank)) {
        ++stats.demandHitsOverride;
        bi->overridden = true;
    } else {
        ++stats.demandHitsNoOverride;
    }



    DPRINTF(LLBP, "LLBP::%s(pc=%#llx): cid=%lx, bcid=%lx, wi=%i, Base:[Hit=%i,p=%i] LLBP:[Hit=%i,p=%i] overridden=%s\n",
        __func__,
        pc,
        bi->cid, bi->bcid, bi->wi,
        tage_bi->hitBank, bi->base_pred,
        bi->hitIndex, bi->llbp_pred,
        bi->overridden);


    auto pred = bi->base_pred;
    if (bi->overridden) {
        // Override the base prediction
        pred = bi->llbp_pred;

        // Also the internally cached prediction information
        tage_bi->tagePred = bi->llbp_pred;
        tage_bi->longestMatchPred = tage_bi->altTaken = bi->llbp_pred;
        tage_bi->hitBank = tage_bi->altBank = 0;
        tage_bi->provider = TAGEBase::BIMODAL_ONLY;
        scltage_bi->lpBranchInfo->predTaken = bi->llbp_pred;
        scltage_bi->lpBranchInfo->loopPredUsed = false;
        scltage_bi->scBranchInfo->usedScPred = false;
    }

    return pred;
}

void
LLBP::squash(ThreadID tid, void *&bp_history)
{
    LLBPBranchInfo *bi = static_cast<LLBPBranchInfo *>(bp_history);
    if (bi->overridden) {
        stats.squashedOverrides++;
    }
    base->squash(tid, bi->ltage_bi);
    squashRCR(tid, bi);

    if (bi->lock_pbe) {
        patternBuffer.squash(bi->cid);
    }
    delete bi;
    bp_history = nullptr;
}

void
LLBP::update(ThreadID tid, Addr pc, bool taken, void *&bp_history, bool resteer,
             const StaticInstPtr &inst, Addr target)
{
    assert(bp_history);
    LLBPBranchInfo *bi = static_cast<LLBPBranchInfo *>(bp_history);
    TAGE_SC_L::TageSCLBranchInfo *tage_bi = static_cast<TAGE_SC_L::TageSCLBranchInfo *>(bi->ltage_bi);

    if (resteer) {
        // For resteers (squashes) we have to correct the RCR
        // and correct TAGE speculative histories
        updateRCR(tid, pc, inst, taken, true, bi);

        base->update(tid, pc, taken, tage_bi, resteer, inst, target);
        return;
    }

    // This is a bit a hackish way to communicate the LLBP override information
    // to the base predictor. The base predictor will use the current bi
    // in its update and allocation functions
    curUpdateBi = bi;
    assert(!resteer);

    // Do the base predictor update.
    base->update(tid, pc, taken, tage_bi, false, inst, target);

    // Update the LLBP for conditional branches
    if (inst->isCondCtrl()) {
        llbpUpdate(tid, pc, taken, bi);
    }


    DPRINTF(LLBP, "LLBP::%s(pc=%llx, inst=%s, taken=%i): "
            "ccid=%llu, uncond=%i\n",
            __func__,
            pc, inst->getName().c_str(),
            taken,
            bi->cid,
            inst->isUncondCtrl()
        );

    // If speculation is disabled update the RCR now
    if (!speculativeHistUpdate) {
        updateRCR(tid, pc, inst, taken, false, bi);
    }

    if (bi->lock_pbe) {
        patternBuffer.commit(bi->cid);
    }

    delete tage_bi;
    delete bi;
    bp_history = nullptr;
}

void
LLBP::branchPlaceholder(ThreadID tid, Addr pc,
                         bool uncond, void * &bpHistory)
{
    LLBPBranchInfo *bi = new LLBPBranchInfo(pc, !uncond);
    base->branchPlaceholder(tid, pc, uncond, bi->ltage_bi);
    bpHistory = (void*)(bi);
}



void
LLBP::calculateKeys(Addr pc, int wi, LLBPBranchInfo *&bi)
{
    for (int i = 1; i <= base->getNumHistoryTables(); i++) {
        if (fltTables[bi->wi][i] < 0) continue;

        uint64_t tag = base->tage->gtag(0, pc, i);
        uint64_t index = base->tage->gindex(0, pc, i);

        // Align the index to the upper bits of the key
        // 10 bits is the number of bits used in the TAGEBase::gindex
        index <<= uint64_t(TTWidth - 10 - 1);
        uint64_t key = (tag ^ index) & ((1ULL << uint64_t(TTWidth)) - 1ULL);
        bi->keys[i] = uint64_t(key) << 10ULL | uint64_t(fltTables[wi][i]);
    }
    bi->keys_valid = true;
}


void
LLBP::llbpPredict(ThreadID tid, Addr pc, LLBPBranchInfo* &bi)
{
    auto tage_bi = static_cast<TAGE_SC_L::TageSCLBranchInfo*>(bi->ltage_bi)->tageBranchInfo;

    // Check if the context depth is adapted
    bi->bcid = rcr.getBaseCtx();
    if (adaptCtxDepth) {
        auto ci = ctt.get(bi->bcid);
        bi->wi = ci ? ci->wi : 0;
    }

    bi->cid = rcr.getCCID(pc, WS[bi->wi]);
    bi->cids[0] = rcr.getCCID(pc, WS[0]);
    bi->cids[nW-1] = rcr.getCCID(pc, WS[nW-1]);

    bi->hitIndex = 0;
    bi->llbp_pred = false;

    // Calculate all keys
    calculateKeys(pc, bi->wi, bi);


    DPRINTF(LLBP,"%llx L2Predict: %lx, CLK=%u, Key: 6:%i, GI:%i, i:%i\n", pc, bi->cid, uint(curCycle()),
            bi->keys[6], tage_bi->tableIndices[6], base->tage->gindex(tid, pc, 6));

    auto context = llbpStorage.get(bi->cid);

    if (!context) {
        ++stats.demandMissesContextUnknown;
        ++stats.demandMissesTotal;
        return;
    }

    // A context exists. Make a prediction
    stats.contextHits++;

    bi->prefetched = false;
    PatternBufferEntry* pbe = patternBuffer.get(bi->cid);
    if (pbe) {
        DPRINTF(LLBP,"Hit in PB: T=%u -> ready=%i\n", uint(pbe->readyTime), pbe->readyTime <= curCycle());
        if (pbe->readyTime <= curCycle()) {
            bi->prefetched = true;
            pbe->usedForPrediction();
        }
        bi->ctx_hit = true;
    }

    // Perform the actual pattern matching
    for (int i = base->getNumHistoryTables(); i > 0; i--) {

        // Skip over unused history lengths
        if (fltTables[bi->wi][i] < 0) continue;

        if (context->patterns.get(bi->keys[i])) {
            bi->hitIndex = i;
            break;
        }
    }

    if (bi->hitIndex > 0) {
        stats.patternHits++;
    }
    DPRINTF(LLBP,"Context=%lx, Pattern=%i, PB hit=%i, prefetched=%i\n", bi->cid, bi->hitIndex, pbe!=nullptr, bi->prefetched);

    // If the tag matches and its prefetches we can use the prediction
    if ((bi->hitIndex > 0) && (bi->prefetched || optimalPrefetching)) {

        bi->hit = true;
        uint64_t key = bi->keys[bi->hitIndex];
        auto &pattern = *context->patterns.get(key);

        patternBuffer.useForPrediction(bi->cid);
        stats.predictionsInFlight.sample(patternBuffer.numInflights(0));
        bi->lock_pbe = true;

        ++stats.demandHitsTotal;
        bi->llbp_pred = pattern.counter >= 0;

        DPRINTF(LLBP, "LLBPHit:%i, CID=%llx, K:%llx,p%i\n",
                bi->hitIndex, bi->cid, key, bi->llbp_pred);
    } else {
        // No pattern found it the PB
        ++stats.demandMissesTotal;

        // Identify whether its a prefetch related miss or
        // a tag miss
        if (bi->hitIndex > 0) {
            if (pbe) {
                ++stats.demandMissesPfInflight;
            } else {
                ++stats.demandMissesContextNotPrefetched;
            }
        } else {
            ++stats.demandMissesPatternMiss;
        }
    }
}


void
LLBP::updateHistories(ThreadID tid, Addr pc, bool uncond, bool taken,
                      Addr target, const StaticInstPtr &inst,
                      void *&bp_history)
{
    // This is the speculativ history update path
    LLBPBranchInfo *bi;
    if (bp_history == nullptr) {
        assert(uncond);
        bi = new LLBPBranchInfo(pc, !uncond);
        bp_history = (void*)(bi);
    } else {
        bi = static_cast<LLBPBranchInfo *>(bp_history);
    }
    // Update the histories of the base predictor
    base->updateHistories(tid, pc, uncond, taken, target, inst, bi->ltage_bi);

    // This is the speculative update path. Thus only update the RCR here
    // if speculation is enabled.
    if (speculativeHistUpdate) {
        updateRCR(tid, pc, inst, taken, false, bi);
    }
}


void
LLBP::updateRCR(ThreadID tid, Addr pc, const StaticInstPtr &inst, bool taken, bool resteer,
                      LLBPBranchInfo *&bi)
{
    // If no speculative update is enabled ignore resteers
    if (resteer && !speculativeHistUpdate) {
        assert(bi->rcrBackup.size() == 0);
        return;
    }

    // Upon mispredition `resteer=true` restore the RCR content
    // as it was before this branch was speculated.
    if (resteer) {
        if (bi->rcr_modified) {
            rcr.restore();
            bi->rcr_modified = false;
        }
    }

    // Update the RCR with the current branch
    if (rcr.update(pc >> instShiftAmt, inst, taken)) {
        bi->rcr_modified = true;
        // If the RCR has updated the context ID, we need to
        // check whether we have to prefetch an upcomming context.
        prefetch();
    }


    DPRINTF(LLBP, "LLBP::%s(pc=%#llx, resteer=%i): pcid=%llx, ccid=%llx, uncond=%i\n",
            __func__, pc, resteer, rcr.getPCID(), rcr.getCCID(), inst->isUncondCtrl());
}

void
LLBP::prefetch()
{
    // If the RCR has updated the context ID, we need to
    // check whether we have to prefetch an upcomming context.
    auto base_ctx = rcr.getPBaseCtx();
    int wi = 0;
    if (adaptCtxDepth) {
        auto ci = ctt.get(base_ctx);
        wi = ci ? ci->wi : 0;
    }
    auto pcid = rcr.getPCID(WS[wi]);

    // If there is not context to prefetch no need to prefetch
    if (!llbpStorage.exists(pcid)) {
        return;
    }

    // If already in the PB abort prefetch
    if (patternBuffer.exists(pcid)) {
        stats.prefetchHitInPB++;
        patternBuffer.touch(pcid);
        return;
    }

    if (installInPB(pcid, curCycle() + backingStorageLatency)) {
        stats.prefetchIssued++;
    } else {
        stats.prefetchDroppedLocked++;
    }
}


void
LLBP::squashRCR(ThreadID tid, LLBPBranchInfo *&bi)
{
    if (bi->rcr_modified) {
        rcr.restore();
        bi->rcr_modified = false;
    }
}


LLBP::PatternBufferEntry*
LLBP::installInPB(uint64_t pcid, Cycles when) {
    // Context exists and needs to be prefetched
    // Get a victim entry
    auto victim = patternBuffer.getVictim(pcid);
    DPRINTF(LLBP,"Install in PB: %llx in %u, now=%u\n", pcid, when, curCycle());

    if (victim) {

        // If the entry is locked due to ongoing prefetch. Don't install in
        // PB but in LLBP right away.
        if (victim->locked || (victim->dependants > 0)) {
            return nullptr;
        }

        if (victim->used) {
            stats.prefetchUsed++;
        } else {
            stats.prefetchUnused++;
        }
        if (victim->dirty) stats.patternBufferEvictDirty++;
        else stats.patternBufferEvictClean++;
    }
    // Copy the prefetched pattern set into the PB
    return patternBuffer.insert(pcid, when);
}


void LLBP::llbpAllocate(ThreadID tid, Addr pc, bool taken, LLBPBranchInfo *bi)
{
    /**************************************************
     * Allocation
     *
     * If the branch was mispredicted, we allocate a new pattern
     * with longer history in the context.
     * The pattern with the weakest confidence is replaced.
     */
    auto& alloc_banks = static_cast<LLBP_TAGE_64KB *>(base->tage)->alloc_banks;
    if (alloc_banks.size() == 0) {
        return;
    }

    // Before allocating context check if the the context
    // depth needs to be addapted.
    for (auto hl : alloc_banks)
        adaptContextDepth(bi, hl);

    bool alloc_any = false;
    for (auto hl : alloc_banks) {
        alloc_any |= fltTables[bi->wi][hl] >= 0;
    }
    if (!alloc_any)
        return;


    Context* context = llbpStorage.get(bi->cid);
    PatternBufferEntry* pbe = patternBuffer.get(bi->cid);
    // If the context does not exist we allocate a new one.
    if (!context) {
        if (!unlimited) {
            llbpStorage.sortContexts(bi->cid);
        }
        // Ensure the victim is not in the L2 predictor
        context = llbpStorage.getVictim(bi->cid);
        if (context) {
            ++stats.backingStorageEvictions;
        }
        // Allocate a new context in the LLBP storage.
        context = llbpStorage.allocate(bi->cid, pc);
        pbe = installInPB(bi->cid, curCycle());
        if (!pbe) stats.patternBufferAllocDropped++;

        DPRINTF(LLBP, "LLBP: CTX Alloc:%#llx,\n", bi->cid);
        ++stats.backingStorageInsertions;
    }

    if (pbe) {
        pbe->dirty = true;
    }

    auto tage_bi = static_cast<TAGE_SC_L::TageSCLBranchInfo *>(bi->ltage_bi)->tageBranchInfo;
    for (auto bank : alloc_banks) {
        if (fltTables[bi->wi][bank] >= 0) {
            // uint64_t key = calculateKey(tage_bi, bank, pc, bi->wi);
            uint64_t key = bi->keys[bank];


            Pattern* ptrn = context->patterns.get(key);
            if (ptrn) {
                continue;
            }

            if (!unlimited) {
                context->sortPatters(key);
                // This prevents the current active pattern to be
                // evicted.
                context->patterns.bump(key);

                std::stringstream s;
                auto& ps = context->patterns.getSet(key);
                for (auto p : ps) {
                    s << std::hex << p.first << ",";
                    s << std::dec << int(p.second.counter) << " | ";
                }

                ptrn = context->patterns.getVictim(key);
                if (ptrn) {
                    if (ptrn->useful) {
                        context->usefulPtrns--;
                    }
                    if (!((ptrn->counter == 0) || (ptrn->counter == -1))) {
                        context->confidentPtrns--;
                    }
                    DPRINTF(LLBP, "LLBP Evict: K:%llx -> %s\n", ptrn->tag, s.str());
                }
            }
            ptrn = context->patterns.insert(key);
            ptrn->tag = key;
            ptrn->counter = taken ? 0 : -1;
            ptrn->hit = 0;
            ptrn->useful = 0;
            ptrn->valid = true;

            DPRINTF(LLBP, "LLBP Alloc:%i, wi:%i, ctx:%llx, K:%llx, T:%i, I:%i\n",
                    bank, bi->wi, bi->cid, key, tage_bi->tableTags[bank], tage_bi->tableIndices[bank]);
            ++stats.allocationsTotal;
            stats.wAlloc[bi->wi]++;
        }
    }
}

/**
 * Update LLBP with the real outcome of a branch.
 * It is currently assumed that the PB still contains the
 * corresponding pattern set (no access latency applied).
 * The context is created if it does not exist yet.
 * The pattern is updated / a longer pattern is allocated.
 *
 * @param tid Thread ID
 * @param pc Program counter
 * @param taken Whether the branch was taken
 * @param bi Branch info
 */
void LLBP::llbpUpdate(ThreadID tid, Addr pc, bool taken, LLBPBranchInfo *bi)
{

    DPRINTF(LLBP, "LLBP::%s(pc=%lx, taken=%i) "
            "cid=%llx, bcid=%llx, hitIndex=%d, "
            "prediction=%d, mispred=%d, "
            "llbp_pred=%d, base_pred=%d, "
            "overridden=%i -> %s,%s\n",
            __func__, pc, taken,
            bi->cid, bi->bcid, bi->hitIndex,
            bi->getPrediction(), taken != bi->getPrediction(),
            bi->llbp_pred, bi->base_pred, bi->overridden,
            bi->overridden ? (bi->llbp_pred == taken ? "good" : "bad") : "no",
            bi->overridden ? (bi->llbp_pred == bi->base_pred ? "same" : "diff") : "no"
        );


    // Do the allocation
    llbpAllocate(tid, pc, taken, bi);


    // Update the gem5 statistics for override tracking
    if (bi->hitIndex > 0) {
        stats.hitLen[bi->hitIndex]++;

        if (bi->overridden) {
            stats.providerLen[bi->hitIndex]++;
            stats.provider++;

            // Mispredictions
            if (bi->llbp_pred != taken) {
                stats.mispredLen[bi->hitIndex]++;
            }

            // Categize whether the override was good or bad
            if (bi->llbp_pred == bi->base_pred) {
                if (bi->llbp_pred == taken) {
                    stats.sameCorrect++;
                } else {
                    stats.sameWrong++;
                }
            } else {
                if (bi->llbp_pred == taken) {
                    stats.goodOverride++;
                    stats.wUseful[bi->wi]++;
                } else {
                    stats.badOverride++;
                }
            }
        } else {
            stats.noOverride++;
            if (!bi->prefetched) {
                stats.noPrefetch++;
            }
        }
    }


    // Check whether the branch context is known
    // If not, we create a new context
    Context* context = llbpStorage.get(bi->cid);
    PatternBufferEntry* pbe = patternBuffer.get(bi->cid);
    // If no context exits return
    if (!context) {
        return;
    }

    if (bi->overridden) {
        // uint64_t key = calculateKey(tage_bi, bi->hitIndex, pc, bi->wi);
        uint64_t key = bi->keys[bi->hitIndex];
        LLBP::Pattern* pattern = context->patterns.get(key);

        if (pbe)
            patternBuffer.commit(bi->cid);

        if (!pattern) {
            return;
        }

        if (adaptCtxDepth) {
            ctt.bump(bi->bcid);
        }

        int8_t conf_before = pattern->counter;
        TAGEBase::ctrUpdate(pattern->counter, taken, patternCounterWidth);
        int8_t conf_after = pattern->counter;

        DPRINTF(LLBP, "LLBP::%s() CID=%llx key=%llx: %d -> %d (%s)\n",
                __func__, bi->cid, key, conf_before, conf_after, taken ? "taken" : "not taken");
        // This function updates the context replacement counter
        // - If a pattern becomes confident (correct prediction)
        //   the replacement counter is increased
        // - If a pattern becomes low confident (incorrect prediction)
        //   the replacement counter is decreased
        if (pattern->counter == (taken ? 1 : -2)) {
            // Context is now medium confidence
            TAGEBase::unsignedCtrUpdate(context->confidence, true,
                                        contextCounterWidth);
            DPRINTF(LLBP,"Ctx:%lx, ConfPtrn:%i\n", bi->cid, context->confidentPtrns);
            if (adaptCtxDepth && (context->confidentPtrns == trackingThreshold)) {
                auto ci = ctt.insert(bi->bcid);
                if (ci->fullPatternSets < 2)
                    ci->fullPatternSets++;
            }
            context->confidentPtrns++;
        }
        else if (pattern->counter == (taken ? -1 : 0)) {
            // Context is now low confidence
            TAGEBase::unsignedCtrUpdate(context->confidence, false,
                                        contextCounterWidth);

            if (adaptCtxDepth && (context->confidentPtrns == trackingThreshold-1)) {
                auto ci = ctt.insert(bi->bcid);
                // stats.citDec++;
                if (ci->fullPatternSets > 0)
                    ci->fullPatternSets--;
            }
            context->confidentPtrns--;
        }
        // Mark the entry as dirty if the content has changed
        if (pbe) {
            pbe->usedForRetBranch();
            if(conf_before != conf_after) {
                pbe->dirty = true;
            }
        }
    }
}



void
LLBP::adaptContextDepth(LLBPBranchInfo *bi, int histLen)
{
    if (!adaptCtxDepth)
        return;

    // ---- Context adaption
    ContextInfo* ci = ctt.get(bi->bcid);
    if (!ci) return;

    DPRINTF(LLBP, "CIT [%lx]: FPS=%i, AvD=%i Wi=%i\n",
            bi->bcid, ci->fullPatternSets, ci->allocVsDrop, ci->wi);
    if ((ci->fullPatternSets >= 0) &&
        (ci->wi < nW-1)) {

        if (ci->allocVsDrop > adaptThreshold) {
            if (ci->wi == 0) {
                ci->wi = nW-1;
                ci->allocVsDrop = 0;
            } else if (ci->wi == nW-1) {
                ci->wi = 0;
                ci->allocVsDrop = 0;
            }
        }
    }


    ////////////////////////////////////////////

    if (histLen < histLenThreshold) {

        if(ci->wi == 0) {
            if (ci->allocVsDrop > 0) {
                ci->allocVsDrop--;
            }
        }

        if (ci->wi == nW-1) {
            ci->allocVsDrop++;
        }
    } else {

        if (ci->wi == nW-1) {
            if (ci->allocVsDrop > 0) {
                ci->allocVsDrop--;
            }
        }

        if (ci->wi == 0) {
            ci->allocVsDrop++;
        }

    }
    bi->cid = bi->cids[bi->wi];
}



/* from LLBP source code: */
LLBP::RCR::RCR(int _T, int _W, int _D, int _shift, int _CTWidth, int _BCTWidth)
    : CTWidth(_CTWidth), BCTTWidth(_BCTWidth),
        T(_T), W(_W), D(_D), S(_shift)
{
    assert((W + D) < maxwindow || "RCR: maxwindow must be larger than W + D");
    bb.resize(maxwindow);
    ctxs = {0, 0, 0, 0};
}

/**
 * Given {n} number of branches starting from the end of the RCR (front of the vec)
 * (minus {skip} # of branches) we create the hash function by shifting
 * each PC by {shift} number if bits i.e.
 *
 *   000000000000|  PC  |    :vec[end-skip]
 * ^ 0000000000|  PC  |00    :vec[end-skip-1]
 * ^ 00000000|  PC  |0000    :vec[end-skip-2]
 *           .                     .
 *           .                     .
 *           .                     .
 * ^ |  PC  |000000000000    :vec[end-skip-n-1]
 * ----------------------
 *       final hash value
 *  Then, the hash value is wrapped to the size of the context tag:
 *  @return final hash value % 2^tagWidthBits
*/
uint64_t
LLBP::RCR::calcHash(int n, int skip, int shift)
{
    uint64_t hash = 0;
    if (bb.size() < (skip + n)) {
        return 0;
    }

    // Compute the rolling hash in element order (newer branches at the front)
    uint64_t sh = 0;
    auto it = bb.begin();
    std::advance(it, skip);
    for (; (it != bb.end()) && (n > 0); it++, n--) {
        uint64_t val = *it;

        // Shift the value
        hash ^= val << uint64_t(sh);

        sh += shift;
        if (sh >= CTWidth) {
            sh -= uint64_t(CTWidth);
        }
    }
    return hash;
}

uint64_t LLBP::RCR::getCCID() {
    return ctxs.ccid & ((1ULL << uint64_t(CTWidth)) - 1);
}

uint64_t LLBP::RCR::getCCID(uint64_t pc, int w) {
    return calcHash(w, D, S)
         & ((1ULL << uint64_t(CTWidth)) - 1);
}
uint64_t LLBP::RCR::getPCID(int w) {
    return calcHash(w, 0, S)
         & ((1ULL << uint64_t(CTWidth)) - 1);
}

uint64_t LLBP::RCR::getPCID() {
    return ctxs.pcid & ((1ULL << uint64_t(CTWidth)) - 1);
}

uint64_t LLBP::RCR::getBaseCtx() {
    return ctxs.bcid & ((1ULL << uint64_t(BCTTWidth)) - 1);
}

uint64_t LLBP::RCR::getPBaseCtx() {
    return ctxs.pbcid & ((1ULL << uint64_t(BCTTWidth)) - 1);
}


bool LLBP::RCR::update(Addr pc, const StaticInstPtr &inst, bool taken)
{
    bool update = false;

    switch (T)
    {
    case 0: // All branches
        update = true;
        break;

    case 1: // Only calls
        if (inst->isCall()) update = true;
        break;

    case 2: // Only calls and returns
        if (inst->isCall() || inst->isReturn())
            update = true;
        break;

    case 3: // Only unconditional branches
        if (inst->isUncondCtrl()) update = true;
        break;

    case 4: // All taken branches
        if (taken) update = true;
        break;
    }

    if (update) {
        // Add the new branch to the history
        bb.push_front(pc);

        // Remove the oldest branch
        if (bb.size() > maxwindow) {
            bb.pop_back();
        }

        // The current context.
        ctxs.ccid = calcHash(W, D, S);
        // The prefetch context.
        ctxs.pcid = calcHash(W, 0, S);
        // The base context.
        #define NBASEH 2
        ctxs.bcid = calcHash(NBASEH, D, S);
        ctxs.pbcid = calcHash(NBASEH, 0, S);

        return true;
    }

    return false;
}



void
LLBP::RCR::restore()
{

    if (bb.size() > 0) {
        bb.pop_front();
    } else {
        warn("BB empty while modified flag is set @ %lu\n", curTick());
    }

    // Because the RCR has changed update all context hashes.
    ctxs.ccid = calcHash(W, D, S);
    ctxs.pcid = calcHash(W, 0, S);
    ctxs.bcid = calcHash(NBASEH, D, S);
    ctxs.pbcid = calcHash(NBASEH, 0, S);
}


LLBP::LLBPStats::LLBPStats(LLBP *llbp)
    : statistics::Group(llbp),
      parent(llbp),
      ADD_STAT(allocationsTotal, statistics::units::Count::get(),
              "Total number of new patterns allocated in any pattern set"),
      ADD_STAT(prefetchIssued, statistics::units::Count::get(),
              "Number of prefetches issued to the backing storage"),
      ADD_STAT(prefetchHitInPB, statistics::units::Count::get(),
              "Number of prefetches dropped because already in the PB"),
      ADD_STAT(prefetchDroppedLocked, statistics::units::Count::get(),
              "Number of prefetches dropped it cannot be installed in the PB"),
      ADD_STAT(prefetchUsed, statistics::units::Count::get(),
              "Number of prefetches used for prediction"),
      ADD_STAT(prefetchUnused, statistics::units::Count::get(),
              "Number of prefetches never used before eviction from the PB"),
      ADD_STAT(prefetchAccuracy, statistics::units::Count::get(),
              "Prefetch accuracy"),
      ADD_STAT(prefetchCoverage, statistics::units::Count::get(),
              "Prefetch coverage"),
      ADD_STAT(contextHits, statistics::units::Count::get(),
              "Total number context hits in the pattern store (without prefetching)"),
      ADD_STAT(patternHits, statistics::units::Count::get(),
              "Total number pattern hits in the pattern store (without prefetching)"),
      ADD_STAT(demandHitsTotal, statistics::units::Count::get(),
              "Total on-demand hits to the pattern buffer"),
      ADD_STAT(demandHitsOverride, statistics::units::Count::get(),
              "On-demand hits to the pattern buffer with LLBP overriding the base predictor"),
      ADD_STAT(demandHitsNoOverride, statistics::units::Count::get(),
              "On-demand hits to the pattern buffer, using the base predictor (LLBP dropped)"),
      ADD_STAT(demandMissesTotal, statistics::units::Count::get(),
              "Total on-demand misses to the pattern buffer"),
      ADD_STAT(demandMissesPatternMiss, statistics::units::Count::get(),
              "On-demand misses to the pattern buffer, the chosen pattern-set did not contain the needed pattern"),
      ADD_STAT(demandMissesContextTooLate, statistics::units::Count::get(),
              "On-demand misses to the pattern buffer where the context was still delayed from insertion latency"),
      ADD_STAT(demandMissesContextNotPrefetched, statistics::units::Count::get(),
              "On-demand misses to the pattern buffer where the context was not scheduled for insertion"),
      ADD_STAT(demandMissesContextUnknown, statistics::units::Count::get(),
              "On-demand misses to the pattern buffer where the context was not in the backing storage"),
      ADD_STAT(demandMissesPfInflight, statistics::units::Count::get(),
              "On-demand misses to the pattern buffer where the context was not scheduled for insertion"),
      ADD_STAT(predictionsInFlight, statistics::units::Count::get(),
              "Number of speculative predictions in flight at the same time"),
      ADD_STAT(patternBufferEvictClean, statistics::units::Count::get(),
              "Number of pattern sets evicted clean from the pattern buffer."),
      ADD_STAT(patternBufferEvictDirty, statistics::units::Count::get(),
              "Number of pattern sets evicted dirty from the pattern buffer. Must be written back"),
      ADD_STAT(patternBufferAllocDropped, statistics::units::Count::get(),
              "Number of pattern sets evicted dirty from the pattern buffer. Must be written back"),
      ADD_STAT(backingStorageEvictions, statistics::units::Count::get(),
              "Number of pattern sets evicted from the backing storage due to capacity limits"),
      ADD_STAT(backingStorageInsertions, statistics::units::Count::get(),
              "Number of pattern sets inserted into the backing storage (including replacements)"),
      ADD_STAT(squashedOverrides, statistics::units::Count::get(),
              "Number of branches predicted by LLBP, but squashed before the outcome was known"),
      ADD_STAT(hitLen, statistics::units::Count::get(),
              "History length of patterns hit in LLBP (distribution)"),
      ADD_STAT(providerLen, statistics::units::Count::get(),
              "History length of LLBP patterns that provided a prediction (distribution)"),
      ADD_STAT(mispredLen, statistics::units::Count::get(),
              "History length of LLBP pattern that caused a misprediction (distribution)"),
      ADD_STAT(provider, statistics::units::Count::get(),
              "Number of times LLBP provided a prediction (committed)"),
      ADD_STAT(noOverride, statistics::units::Count::get(),
              "Number of times LLBP hit in the PB but did now overrride (committed)"),
      ADD_STAT(noPrefetch, statistics::units::Count::get(),
              "Number of times LLBP did not override because it was not prefeched (committed)"),
      ADD_STAT(sameCorrect, statistics::units::Count::get(),
              "Number of times LLBP and TAGE where both correct (committed)"),
      ADD_STAT(sameWrong, statistics::units::Count::get(),
              "Number of times LLBP and TAGE where both wrong (committed)"),
      ADD_STAT(goodOverride, statistics::units::Count::get(),
              "Number of times the LLBP override was good (TAGE would had been wrong)"),
      ADD_STAT(badOverride, statistics::units::Count::get(),
              "Number of times the LLBP override was bad (TAGE would had been correct)"),
      ADD_STAT(wAlloc, statistics::units::Count::get(),
              "Context depth of new pattern allocations"),
      ADD_STAT(wUseful, statistics::units::Count::get(),
              "Context depth of useful predictions")

{
    hitLen.init(40).flags(statistics::pdf);
    providerLen.init(40).flags(statistics::pdf);
    mispredLen.init(40).flags(statistics::pdf);
    wAlloc.init(6).flags(statistics::pdf);
    wUseful.init(6).flags(statistics::pdf);

    assert(parent);

    predictionsInFlight.init(0, 64, 8)
                       .flags(statistics::pdf);

    prefetchAccuracy = prefetchUsed / prefetchIssued;
    prefetchCoverage = demandHitsTotal / patternHits;
}

bool
LLBP_TAGE_64KB::tagePredict(ThreadID tid, Addr pc,
              bool cond_branch, TAGEBase::BranchInfo* bi)
{
    bool pred = TAGE_SC_L_TAGE_64KB::tagePredict(tid, pc, cond_branch, bi);
    return pred;
}

void
LLBP_TAGE_64KB::handleAllocAndUReset(bool alloc, bool taken, TAGEBase::BranchInfo* bi, int nrand)
{
    alloc_banks.clear();

    // If LLBP has overridden the base predictor and a misprediction occured
    // we need to let the base predictor know which length of the history
    // has been matched.
    bool modified = false;
    if (parent->curUpdateBi->overridden) {
        // If LLBP was provider we allocate if the prediction was wrong
        // and the history length is shorter than the maximum.
        alloc = (parent->curUpdateBi->llbp_pred != taken) && (parent->curUpdateBi->hitIndex < nHistoryTables);
        bi->hitBank = parent->curUpdateBi->hitIndex;
        modified = true;
    }
    // Do the actual allocation
    TAGE_SC_L_TAGE_64KB::handleAllocAndUReset(alloc, taken, bi, nrand);

    // Afterwards, reset the override otherwise the base predictor
    // will update an incorrect entry
    if (modified) {
        bi->hitBank = 0;
    }
}

int
LLBP_TAGE_64KB::allocateEntry(int bank, TAGEBase::BranchInfo* bi, bool taken)
{
    auto r = TAGE_SC_L_TAGE_64KB::allocateEntry(bank, bi, taken);

    // If the allocation was successful, record the table bank such that
    // LLBP can allocate a new pattern with the same history length.
    if (r > 0) {
        alloc_banks.push_back(bank);
    }
    return r;
}

bool
LLBP_TAGE_64KB::tageCorrect(bool taken, TAGEBase::BranchInfo* bi)
{
    return  (bi->hitBank > 0) ? (bi->longestMatchPred == taken) :
            (bi->altTaken == taken);
}

void
LLBP_TAGE_64KB::handleTAGEUpdate(Addr pc, bool taken, TAGEBase::BranchInfo* bi)
{
    // Update the usefulness
    if (parent->curUpdateBi->hitIndex > 0) {
        // If TAGE was provider, it was correct and
        // LLBP was incorrect this prediction was useful.
        bool prim_correct = tageCorrect(taken, bi);
        if ((!parent->curUpdateBi->overridden) && prim_correct && (parent->curUpdateBi->llbp_pred != taken)) {
            if (bi->hitBank > 0) {
                if(gtable[bi->hitBank][bi->hitBankIndex].u < ((1 << tagTableUBits) -1)) {
                    gtable[bi->hitBank][bi->hitBankIndex].u++;
                }
            }
        }
    }
    // Update the BIM if the LLBP prediction was wrong.
    if ((parent->curUpdateBi->overridden) && (parent->curUpdateBi->llbp_pred != taken)){
        TAGE_SC_L_TAGE_64KB::baseUpdate(pc, taken, bi);
    }
    // Only update the providing component if LLBP has overridden than
    // don't update TAGE.
    if (parent->curUpdateBi->overridden) {
        return;
    }
    // If not overridden do the normal TAGE update
    TAGE_SC_L_TAGE_64KB::handleTAGEUpdate(pc, taken, bi);
}

bool
LLBP_TAGE_64KB::isUseful(bool taken, TAGEBase::BranchInfo* bi) const
{
    // If LLBP overrides we do the usefulness update in `handleTAGEUpdate`
    return (parent->curUpdateBi->hitIndex > 0) ? false
           : TAGE_SC_L_TAGE_64KB::isUseful(taken, bi);
}

bool
LLBP_TAGE_64KB::isNotUseful(bool taken, TAGEBase::BranchInfo* bi) const
{
    return (parent->curUpdateBi->hitIndex > 0) ? false
           : TAGE_SC_L_TAGE_64KB::isNotUseful(taken, bi);
}

} // namespace branch_prediction
} // namespace gem5
