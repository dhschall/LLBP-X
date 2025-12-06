/**
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
 */

#ifndef __CPU_PRED_LLBP_HH__
#define __CPU_PRED_LLBP_HH__

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/cache/associative_cache.hh"
#include "base/cache/cache_entry.hh"
#include "base/statistics.hh"
#include "base/types.hh"
#include "cpu/pred/tage_sc_l.hh"
#include "cpu/pred/tage_sc_l_64KB.hh"
#include "params/LLBP.hh"
#include "params/LLBP_TAGE_64KB.hh"
#include "cpu/pred/cache.h"
#include "sim/clocked_object.hh"

namespace gem5
{

namespace branch_prediction
{

class LLBP : public ConditionalPredictor, public Clocked
{
  public:
    LLBP(const LLBPParams &params);

    bool lookup(ThreadID tid, Addr pc, void * &bp_history) override;

    void squash(ThreadID tid, void * &bp_history) override;
    void update(ThreadID tid, Addr pc, bool taken,
                void * &bp_history, bool squashed,
                const StaticInstPtr & inst, Addr target) override;

    void init() override;
    void branchPlaceholder(ThreadID tid, Addr pc,
                        bool uncond, void * &bpHistory) override;

    void updateHistories(ThreadID tid, Addr pc, bool uncond,
                         bool taken, Addr target,
                         const StaticInstPtr &inst,
                         void * &bp_history) override;
  protected:

    TAGE_SC_L* base;


    struct LLBPBranchInfo {
        const Addr pc;
        const bool conditional;

        bool hit = false;
        bool overridden = false;
        bool llbp_pred = false;
        bool base_pred = false;
        int hitIndex = 0; // TODO: rename this to hitIndex of llbpHit
        uint64_t key = 0;
        uint64_t keys[40];
        bool keys_valid = false;
        std::list<uint64_t> rcrBackup;
        bool rcr_modified = false;
        uint64_t cid = 0;
        uint64_t cids[6];
        uint64_t bcid = 0;
        int wi = 0;
        bool adapt = false;
        void* ltage_bi = nullptr;
        bool ctx_hit = false;
        bool lock_pbe = false;
        bool prefetched = false;

        LLBPBranchInfo(Addr pc, bool conditional)
          : pc(pc),
            conditional(conditional)
        {}

        bool getPrediction() {
            return overridden ? llbp_pred : base_pred;
        }

        ~LLBPBranchInfo()
        {}
    };

  public:
    // The branch info of the branch that currently gets updated.
    // A bit of a hack to communicate the LLBP prediction information
    // to the base predictor.
    LLBPBranchInfo *curUpdateBi = nullptr;


  protected:

    /********************************************************************
     * LLBP Pattern
     *
     * Consists of the history length field and the tag.
     * In the model we concatenate both to form a key.
     * key = (tag << 10) | length
     * This simplifies model complexity
     ******************************************************************/
    struct Pattern {
        uint64_t tag = 0;
        int8_t counter = 0;
        int hit = 0;
        int useful = 0;
        bool valid = false;
    };


    /********************************************************************
     * Pattern Set
     *
     * The pattern sets are implemented as set associative cache. The
     * lower bits of the key - to lookup a pattern in the pattern set
     * - are used for the history length which realizes the four way
     * associativity. In the constructor we assign each history an
     * index
     ******************************************************************/
    struct PatternSet : public BaseCache<uint64_t, Pattern>{
        PatternSet(size_t max_size, size_t assoc) :
            BaseCache<uint64_t, Pattern>(max_size, assoc)
        {}

        Pattern* insert(const uint64_t &key) {
            return BaseCache<uint64_t, Pattern>::insert(key);
        }
    };


    /********************************************************************
     * Program Context
     *
     * A program context contains one pattern set and is indexed by
     * a key formed by hashing W unconditional branches.
     * This struct contains some additional meta data for replacement
     * and statistics.
     ********************************************************************/
    struct Context {
        bool valid = false;
        uint64_t key = 0;
        uint64_t pc = 0;
        int correct = 0;
        int incorrect = 0;
        int useful = 0;
        int conflict = 0;
        uint replace = 0; // @todo clean up
        int ctr = 0;
        int usefulPtrns = 0;
        int confidentPtrns = 0;
        int W = 0;
        uint8_t confidence = 0;

        // The contexts pattern set.
        PatternSet patterns;

        Context(uint64_t k, uint64_t p, int n, int assoc)
          : valid(true), key(k), pc(p),
            patterns(n, assoc)
        {}

        // Before a pattern in the pattern set is replaced, the patterns are
        // sorted from the highest to the lowest confidence. This is done to
        // determine which pattern should be evicted.
        void sortPatters(const uint64_t key) {
            auto& set = patterns.getSet(key);
            set.sort(
                [](const std::pair<uint64_t, Pattern>& a, const std::pair<uint64_t, Pattern>& b)
                {
                    return abs(2*a.second.counter+1) > abs(2*b.second.counter+1);
                });
        }
    };



    /********************************************************************
     * LLBP Storage
     *
     * LLBPs high-capacity structure to store all pattern sets.
     * It's implemented as a set associative cache.
     * The Context directory (CD) can be thought of as the tag array while the
     * LLBPStorage is the data array. In this simulation model, both LLBP
     * and CD are represented with a single data structure.
     ********************************************************************/
    class LLBPStorage : public BaseCache<uint64_t, Context>{
        typedef typename std::pair<uint64_t, Context> key_value_pair_t;
	    typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;
        const int n_patterns;
        const int _ptrn_assoc;

    public:

        LLBPStorage(int n_ctx, int n_patterns, int ctx_assoc, int ptrn_assoc)
          : BaseCache<uint64_t, Context>(n_ctx, ctx_assoc),
            n_patterns(n_patterns), _ptrn_assoc(ptrn_assoc)
        {
        }

        // This function creates a new context but does not install it.
        Context* createNew(uint64_t key, uint64_t pc) {
            return new Context(key, pc, n_patterns, _ptrn_assoc);
        }

        // This function will allocate a new context for the
        // given key if it does not exist.
        // It Will return the created context.
        // Note that this function will NOT sort the contexts.
        // Therefore, make sure to call the sorting function before
        // this function
        Context* allocate(uint64_t key, uint64_t pc) {

		    auto c = this->get(key);
            if (c != nullptr) {
                return c;
            }

            auto& set = this->getResizedSet(key);

            set.push_front(
                key_value_pair_t(key, Context(key, pc, n_patterns, _ptrn_assoc)));
            _index[key] = set.begin();
            return &set.front().second;
        }

        // Sort the contexts in a set based on the replacement counter.
        void sortContexts(uint64_t key) {
            auto& set = this->getSet(key);
            set.sort(
                [](const key_value_pair_t& a, const key_value_pair_t& b)
                {
                    // return a.second.replace > b.second.replace;
                    return a.second.confidence > b.second.confidence;
                });
        }
    } llbpStorage;



    /********************************************************************
     * Pattern Buffer
     *
     * The pattern buffer is a small set associative cache that maintains
     * the most recent executed pattern set. Upcomming contexts
     * are prefetched into the pattern buffer and predictions are made from
     * the pattern buffer.
     *
     * Note that in the model we don't move the patterns into the pattern
     * buffer. Instead we directly modify the patterns in the LLBPStorage.
     * The pattern buffer models the caching behaviour and is only used
     * in the timing model.
     */

    struct PatternBufferEntry {
        const uint64_t cid;
        Cycles readyTime;
        Cycles lastUsed;
        bool valid;
        bool locked;
        bool dirty;
        bool used;
        bool usedRetired;
        int dependants;

        PatternBufferEntry(uint64_t id, Cycles ready)
          : cid(id), readyTime(ready), lastUsed(ready), valid(true),
            locked(false), dirty(false), used(false), usedRetired(false),
            dependants(0)
        {}

        void usedForPrediction() { used = true; }
        void usedForRetBranch() { usedRetired = true; }
    };

    class PatternBuffer : public BaseCache<uint64_t, PatternBufferEntry> {
      public:
        PatternBuffer(int n, int assoc)
          : BaseCache<uint64_t, PatternBufferEntry>(n, assoc)
        {
        }

        std::unordered_map<uint64_t,int> inFlightPred;

        void removeInFlight(uint64_t cid) {
            auto it = inFlightPred.find(cid);
            if (it == inFlightPred.end()) {
                return;
            }

            it->second--;
            if (it->second <= 0) {
                inFlightPred.erase(it->first);
            }
        }
        void addInFlight(uint64_t cid) {
            inFlightPred[cid]++;
        }

        void useForPrediction(uint64_t cid) {
            auto e = get(cid);
            if (!e) return; // TODO
            assert(e);
            e->used = true;
            e->dependants++;
            e->readyTime = Cycles(0); // TODO hack to prevent inflight squashes.
            touch(cid);
            addInFlight(cid);
        }

        void commit(uint64_t cid) {
            auto e = get(cid);
            if (!e) return; // TODO
            assert(e);
            e->usedRetired = true;
            if (e->dependants > 0) {
                e->dependants--;
            }
            removeInFlight(cid);
        }

        void squash(uint64_t cid) {
            auto e = get(cid);
            if (!e) return;
            if (e->dependants > 0) {
                e->dependants--;
            }
            removeInFlight(cid);
        }

        int numInflights(uint64_t cid) {
            int n = 0;
            for (auto& entry : _index) {
                if (entry.second->second.dependants > 0) {
                    n++;
                }
            }
            return n;
        }



        PatternBufferEntry* insert(uint64_t cid, Cycles now) {
            auto v = get(cid);
            if (v != nullptr) {
                touch(cid);
                return v;
            }
            // Get the set with a free item
            auto& set = getResizedSet(cid);


            set.push_front(key_value_pair_t(cid, PatternBufferEntry(cid, now)));
            assert(set.size() <= _assoc);
            _index[cid] = set.begin();

            assert(_index.size() <= _max_size);

            set.front().second.lastUsed = now;
            set.front().second.valid = true;
            set.front().second.locked = false;
            set.front().second.dirty = false;
            set.front().second.used = false;
            set.front().second.usedRetired = false;

            return &set.front().second;
        }

    } patternBuffer;

    /** Prefetch functions */
    void prefetch();
    PatternBufferEntry* installInPB(uint64_t pcid, Cycles when);

    /** Helper for pattern tag calculation */
    void calculateKeys(Addr pc, int wi, LLBPBranchInfo *&bi);

    /** LLBP prediction function */
    void llbpPredict(ThreadID tid, Addr pc, LLBPBranchInfo* &bi);

    /** The LLBP update and allocation functions */
    void llbpUpdate(ThreadID tid, Addr pc, bool taken, LLBPBranchInfo* bi);
    void llbpAllocate(ThreadID tid, Addr pc, bool taken, LLBPBranchInfo *bi);

    /** Updated functions for the RCR */
    void updateRCR(ThreadID tid, Addr pc, const StaticInstPtr &inst,
                   bool taken, bool resteer, LLBPBranchInfo *&bi);
    void squashRCR(ThreadID tid, LLBPBranchInfo *&bi);


    /** Access latency to the pattern store */
    const Cycles backingStorageLatency;

    /** Pattern set configuration */
    const int patternSetCapacity;
    const int patternSetAssoc;

    /** Counter widths */
    const int contextCounterWidth;
    const int patternCounterWidth;

    /** Pattern tag size */
    const int TTWidth;

    /** Performs ideal look ups right into the pattern store */
    const bool optimalPrefetching;

    /** Speculative RCR updating. This parameter is set depending on
     * the base predictor */
    bool speculativeHistUpdate;

    /** Flag to disable sorting before eviction to simulate unlimited
     * capacity */
    const bool unlimited;

    /********************************************************************
     * LLBP-X related functionality
     ********************************************************************
     * At the heard of LLBP-X is the Context Tracking Table (CTT).
     * It is the structure used by LLBP-X to track contexts with
     * high utilization.
     * If the number of usful patterns exceed the `trackingThreshold`
     * a context gets installed into the CTT. And the CTT tracks the
     * allocation length. A counter is updated at every allocation
     * depending on whether the history length is longer or shorter
     * than the `histLenThreshold`.
     * Adaption happens once the counter exceeds the `adaptThreshold`
     ********************************************************************/
    struct ContextInfo {
        int correct = 0;
        int incorrect = 0;
        int wi = 0;
        int fullPatternSets = 0;
        int allocVsDrop = 0;
    };

    struct CtxInfoTable : public BaseCache<uint64_t, ContextInfo>{
        CtxInfoTable(size_t max_size, size_t assoc) :
            BaseCache<uint64_t, ContextInfo>(max_size, assoc)
        {}
    } ctt;


    /** Enable context depth adaption (LLBP-X) */
    const bool adaptCtxDepth;

    /** Threshold bejond which the CTT starts tracking a context */
    const int trackingThreshold;

    /** Threshold bejond which the CTT switches to a deeper context */
    const int adaptThreshold;

    /** History length threshold determining whether to increment or decrement
     * the adaption counter in the CTT.*/
    const int histLenThreshold;

    /** The different context depths */
    static const int nW = 2;
    const int WS[nW] = {2,64};
    std::vector<int> fltTables[nW];
    std::unordered_map<int,int> WSr;

    /** Function to adapth the context */
    void adaptContextDepth(LLBPBranchInfo *bi, int hlen);



     /********************************************************************
     * Rolling Context Register (RCR)
     *
     * Maintains a history of previous fetched unconditional branches (UBs)
     * Computes the context ID's.
     * The RCR is updated speculative and drives the prefetching of pattern
     * sets.
     */
    class RCR {
    public:
        const int maxwindow = 400;

        uint64_t calcHash(int n, int start=0, int shift=0);

        // The context tag width
        const int CTWidth;
        // The base context tag width
        const int BCTTWidth;

        // A list of previouly taken branches
        std::list<uint64_t> bb;

        // We compute the context ID and prefetch context ID
        // only when the content of the RCR changes.
        struct {
            uint64_t ccid = 0;
            uint64_t pcid = 0;
            uint64_t bcid = 0;
            uint64_t pbcid = 0;
        } ctxs;


    public:
        // The hash constants
        const int T, W, D, S;

        RCR(int _T, int _W, int _D, int _shift, int _CTWidth, int _BCTWidth);

        // Push a new branch into the RCR.
        bool update(Addr pc, const StaticInstPtr & inst, bool taken);

        // Restore the RCR state from a list
        void restore();

        // Get the current context ID
        uint64_t getCCID();
        uint64_t getCCID(uint64_t pc);
        uint64_t getCCID(uint64_t pc, int w);

        // Get the prefetch context ID
        uint64_t getPCID(int w);
        uint64_t getPCID();

        uint64_t getBaseCtx();
        uint64_t getPBaseCtx();
    } rcr;

    struct LLBPStats : public statistics::Group
    {
        LLBPStats(LLBP *llbp);

        LLBP* parent;

        statistics::Scalar allocationsTotal;
        statistics::Scalar prefetchNoContext;
        statistics::Scalar prefetchIssued;
        statistics::Scalar prefetchHitInPB;
        statistics::Scalar prefetchDroppedLocked;
        statistics::Scalar prefetchSquashed;
        statistics::Scalar prefetchUsed;
        statistics::Scalar prefetchUnused;
        statistics::Formula prefetchAccuracy;
        statistics::Formula prefetchCoverage;
        statistics::Scalar contextHits;
        statistics::Scalar patternHits;
        statistics::Scalar demandHitsTotal;
        statistics::Scalar demandHitsOverride;
        statistics::Scalar demandHitsNoOverride;
        statistics::Scalar demandMissesTotal;
        statistics::Scalar demandMissesPatternMiss;
        statistics::Scalar demandMissesContextTooLate;
        statistics::Scalar demandMissesContextNotPrefetched;
        statistics::Scalar demandMissesContextUnknown;
        statistics::Scalar demandMissesPfInflight;
        statistics::Distribution predictionsInFlight;
        statistics::Scalar patternBufferEvictClean;
        statistics::Scalar patternBufferEvictDirty;
        statistics::Scalar patternBufferAllocDropped;
        statistics::Scalar backingStorageEvictions;
        statistics::Scalar backingStorageInsertions;
        statistics::Scalar squashedOverrides;
        statistics::Vector hitLen;
        statistics::Vector providerLen;
        statistics::Vector mispredLen;
        statistics::Scalar provider;
        statistics::Scalar noOverride;
        statistics::Scalar noPrefetch;
        statistics::Scalar sameCorrect;
        statistics::Scalar sameWrong;
        statistics::Scalar goodOverride;
        statistics::Scalar badOverride;

        statistics::Vector wAlloc;
        statistics::Vector wUseful;

   } stats;
};

/**
 * Modified TAGE-SC-L version that allows integration with LLBP
 */
class LLBP_TAGE_64KB : public TAGE_SC_L_TAGE_64KB
{
    LLBP *parent;

    public:
    LLBP_TAGE_64KB(const LLBP_TAGE_64KBParams &p)
      : TAGE_SC_L_TAGE_64KB(p)
    {}
    bool tagePredict(ThreadID tid, Addr branch_pc, bool cond_branch,
                     TAGEBase::BranchInfo* bi) override;
    void handleAllocAndUReset(bool alloc, bool taken,
                              TAGEBase::BranchInfo* bi, int nrand) override;

    void handleTAGEUpdate(Addr branch_pc, bool taken,
                          TAGEBase::BranchInfo* bi) override;
    int allocateEntry(int bank, TAGEBase::BranchInfo* bi, bool taken) override;
    bool isUseful(bool taken, TAGEBase::BranchInfo* bi) const override;
    bool isNotUseful(bool taken, TAGEBase::BranchInfo* bi) const override;

    bool tageCorrect(bool taken, TAGEBase::BranchInfo* bi);

    void setParent(LLBP *p) {
        parent = p;
    }

    std::vector<int> alloc_banks;
};

} // namespace branch_prediction
} // namespace gem5

#endif // __CPU_PRED_LLBP_HH__
