/* MIT License
 *
 * Copyright (c) 2024 David Schall and EASE lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#pragma once

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <unordered_map>
#include <list>
#include <queue>


#include "bpmodels/tage/tage_scl.h"

// #include "utils/fileutils.h"
#include "bpmodels/components/counters.h"
#include "bpmodels/components/cache.h"
#include "utils/histogram.h"

namespace LLBP {


struct LLBPXConfig;

class LLBPX : public TageSCL {

  public:
    LLBPX(LLBPXConfig config);
    ~LLBPX();

    void UpdatePredictor(uint64_t PC, bool resolveDir,
                         bool predDir, uint64_t branchTarget) override;
    void TrackOtherInst(uint64_t PC, OpType opType, bool taken,
                        uint64_t branchTarget) override;
    void PrintStat(double NUMINST) override;
    void tick() override;
    void btbMiss() override;
    void setState(bool warmup) override;

  private:


    // Override some base class functions
    bool predict(uint64_t pc) override;
    void updateTables(uint64_t pc, bool resolveDir, bool predDir) override;
    void updateStats(bool taken, bool predtaken, uint64_t PC) override;
    void updateGHist(const bool bit) override;
    int allocate(int idx, uint64_t pc, bool taken) override;
    void resetStats() override;


    typedef uint64_t Key;
    Key KEY[MAXNHIST];  //


    /********************************************************************
     * LLBP Pattern
     *
     * Consists of the history length field and the tag.
     * In the model we concatenate both to form a key.
     * key = (tag << 10) | length
     * This simplifies model complexity
     ******************************************************************/
    struct Pattern {
      int length;
      uint tag;
      int idx;
      int8_t ctr;
      uint replace;
      bool dir;
      int useful = 0;
      int correct = 0;
      int incorrect = 0;
      Key key = 0;
      int evicted = 0;
      int evicted_ctx = 0;
      uint64_t pc = 0;
      uint64_t base_ctx = 0;
      int W = 0;
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
        uint replace = 0;
        int ctr = 0;
        int usefulPtrns = 0;
        int confidentPtrns = 0;
        int W = 0;

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
                    return abs(center(a.second.ctr)) > abs(center(b.second.ctr));
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
        const int n_contexts;
        const int n_patterns;
        const int _ptrn_assoc;

    public:

        LLBPStorage(int n_ctx, int n_patterns, int ctx_assoc, int ptrn_assoc)
          : BaseCache<uint64_t, Context>(n_ctx, ctx_assoc),
            n_contexts(n_ctx), 
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
                    return a.second.replace > b.second.replace;
                });
        }
    } llbpStorage;



    bool bim_pred;
    unsigned bimConf;



    /********************************************************************
     * Rolling Context Register RCR
     *
     * The RCR maintains the previous executed branches to compute
     * a context ID.
     *
     * The hash function is defined by 4 paramenters
     *
     * T: Type of history (T). Which branches should be hased
     *     0: All branches, 1: Only calls, 2: Calls and returns
     *     3: All unconditional branches, 4: All taken branches
     *
     * W: Number of branches that should be hashed (W in the paper).
     * D: Number of most recent branches skipped for CCID. Adds delay which
     *    is used to prefetch. (D in the paper.)
     * S: Number of bits to shift the PC's. Is useful to avoid ping-pong context
     *    due to the XOR function in case a loop is executed
     *
     * ********************************************************************* *
     * EXAMPLE                                                               *
     *                                                                       *
     *                       pb-index (2.)  (3.)                             *
     *                      v             v     v                            *
     * history buffer : |l|k|j|i|h|g|f|e|d|c|b|a|                            *
     *                            ^prefetch (2.)^                            *
     *                                                                       *
     * a is the newest branch PC added to the buffer, l the oldest.          *
     * (2.) = W = 7; (3.) = D = 3                                            *
     * branches used to obtain PB index hash: j to d                         *
     * branches used to obtain hash to prefetch into PB: g to a              *
     * ********************************************************************* *
     */
    class RCR {
        LLBPX& parent;
        // The maximum number of branches to consider for the context
        const int maxwindow = 200;

        uint64_t calcHash(std::list<uint64_t> &vec, int n, int start=0, int shift=0);

        // The context tag width
        const int CTWidth;
        // The base context tag width
        const int BCTTWidth;

        // A list of previouly taken branches
        std::list<uint64_t> bb[10];

        // We compute the context ID and prefetch context ID
        // only when the content of the RCR changes.
        struct {
            uint64_t ccid = 0;
            uint64_t pcid = 0;
            uint64_t bcid = 0;
            uint64_t pbcid = 0;
        } ctxs;

        int branchCount = 0;

    public:
        // The hash constants
        const int T, W, D, S;

        RCR(LLBPX &_parent, int _T, int _W, int _D, int _shift, int _CTWidth, int _BCTWidth);

        // Push a new branch into the RCR.
        bool update(uint64_t pc, OpType type, bool taken);

        // Get the current context ID
        uint64_t getCCID();
        uint64_t getCCID(uint64_t pc);
        uint64_t getCCID(uint64_t pc, int w);

        // Get the prefetch context ID
        uint64_t getPCID(int w);
        uint64_t getPCID();
        // uint64_t getPCID(uint64_t pc);

        uint64_t getBaseCtx();
        uint64_t getPBaseCtx();
    } rcr;


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
    struct PBEntry {
        Key key;
        bool dirty;
        bool newlyAllocated;
        bool used;
        bool useful;
        int origin;
        bool valid;
        int prefetchtime;
        bool locked;
        PBEntry(Key c)
          : key(c), dirty(false),
            newlyAllocated(false),
            used(false), useful(false),
            origin(0),
            valid(false),
            prefetchtime(0),
            locked(false)
        {}
        PBEntry() : PBEntry(0) {}
    };

    class PatternBuffer : public BaseCache<uint64_t, PBEntry> {
      public:
        PatternBuffer(int n, int assoc)
          : BaseCache<uint64_t, PBEntry>(n, assoc)
        {
        }

        PBEntry* insert(PBEntry &entry) {;
            auto v = get(entry.key);
            if (v != nullptr) {
                return v;
            }

            // Get the set with a free item
            auto& set = getResizedSet(entry.key);

            set.push_front(key_value_pair_t(entry.key, entry));
            _index[entry.key] = set.begin();
            return &set.front().second;
        }
    } patternBuffer;


    // Pointers to context, llbp pattern and PB entry in case of a
    // LLBP pattern/context match.
    Context* HitContext;
    Pattern* llbpEntry;
    PBEntry* pbEntry;


    // A struct to maintain the prediction info from LLBP.
    struct LLBPPredInfo {
        bool hit = false;
        int pVal = 0;
        bool pred = false;
        unsigned conf = 0;
        int histLength = 0;
        bool prefetched = false;
        bool isProvider = false;
        bool shorter = false;
        int wi = 0;
    } llbp;

    // The prediction function for the LLBP.
    void llbpPredict(uint64_t pc);

    // The LLBP update function.
    void llbpUpdate(uint64_t PC, bool resolveDir, bool predDir);

    // The LLBP allocate function.
    bool llbpAllocate(int idx, uint64_t pc, bool taken);

    // Function to allocate a new context.
    Context* allocateNewContext(uint64_t pc, uint64_t key);

    // A map to filter the used history lengths.
    std::unordered_map<int,int> fltTables[6];

    // The number of contexts in the CD/LLBP
    const int numContexts;
    // The number of patterns per pattern set.
    const int numPatterns;
    // Bit width for pattern tag.
    const int TTWidth;
    // Constants for the patterns counter widths
    const int CtrWidth;
    const int ReplCtrWidth;
    const int CtxReplCtrWidth;

    // Folded history register. Same as in the TAGE predictor.
    FoldedHistoryFast* fghrT1[MAXNHIST];
    FoldedHistoryFast* fghrT2[MAXNHIST];


    // Override the chooser functions to arbitrate between
    // the baseline TAGE and LLBP
    bool isNotUseful(bool taken) override;
    bool isUseful(bool taken) override;
    void updateL2Usefulness(bool taken);

    unsigned chooseProvider() override;
    bool useLLBP();

    inline bool llbpCorrect(bool taken);
    inline bool primCorrect(bool taken);
    inline bool tageCorrect(bool taken);
    inline bool llbpUseful(bool taken);


    // Timing simulation --------------------------------------
    // Methods, variables and structures for the prefetching
    // functionality. Prefetching is only modelled if `simulateTiming`
    // is set to true.
    const bool simulateTiming;

    void prefetch();
    void tickPrefetchQueue();
    void squashPrefetchQueue(bool btbMiss=false);
    void installInPB(PBEntry &entry, bool bypass=false);

    // The prefetch queue
    std::list<PBEntry> prefetchQueue;
    bool warmup = false;
    bool do_prefetch = false;
    const int accessDelay;

    int nallocated;
    const int maxAlloc = 2;
    const int adaptThreshold;
    const int histLenThreshold;
    static const int nW = 6;
    const int WS[nW] = {2,4,8,16,32,64};

    const int BCTWidth;
    std::unordered_map<int,int> WSr;
    void evictPattern(Context* ctx, Pattern* ptrn, bool ctx_evict=false);
    struct ContextInfo {
        int correct = 0;
        int incorrect = 0;
        int llbpUseful = 0;
        int llbpProv = 0;
        int tageProv = 0;
        int bimProv = 0;
        int nPatterns = 0;
        int nUsefulPatterns = 0;
        int wi = 0;
        int allocPtrns = 0;
        int totAllocPtrns = 0;
        int allocPtrnsLong = 0;
        int allocPge13 = 0;
        int allocPge15 = 0;
        int allocPge17 = 0;
        int allocCtxs = 0;
        int fullPatternSets = 0;
        int allocDropLonger = 0;
        int allocDropShorter = 0;
        int allocVsDrop = 0;
        uint64_t hlsum = 0;
        uint64_t coverSum = 0;
    };
    std::unordered_map<uint64_t,ContextInfo> baseContexts;


    struct CtxInfoTable : public BaseCache<uint64_t, ContextInfo>{
        CtxInfoTable(size_t max_size, size_t assoc) :
            BaseCache<uint64_t, ContextInfo>(max_size, assoc)
        {}
    } cit;

    // Some histograms
    Histogram<size_t> primMispLength;
    Histogram<size_t> llbpMispLength;
    Histogram<size_t> primProvLength;
    Histogram<size_t> llbpProvLength;
    Histogram<size_t> numHistPerContext;
    Histogram<size_t> numUsefulHistPerContext;

    Histogram<size_t> wAlloc;
    Histogram<size_t> wUseful;

    // Some stats
    struct l2_stats
    {

        int tageProv = 0;
        int tageCorrect = 0;
        int tageWrong = 0;
        int sclProv = 0;
        int sclCorrect = 0;
        int sclWrong = 0;
        int baseProv = 0;
        int baseCorrect = 0;
        int baseWrong = 0;

        int l2Prov = 0;
        int l2Correct = 0;
        int l2Wrong = 0;
        int l2CtxHit = 0;
        int l2PtrnHit = 0;
        int l2notBecauseShorter = 0;
        int l2notBecauseNotPrefetched = 0;

        int l2cacheDirtyEvict = 0;
        int l2cacheCleanEvict = 0;
        int l2PFHitInQueue = 0;
        int l2PFHitInCache = 0;
        int l2PFHitInCI = 0;

        int pfDroppedLocked = 0;
        int pfDroppedMispredict = 0;
        int pfDroppedBTBMiss = 0;

        int primCorrect = 0;
        int primWrong = 0;
        int l2OverrideGood = 0;
        int l2OverrideBad = 0;
        int l2OverrideSameCorr = 0;
        int l2OverrideSameWrong = 0;
        int l2NoOverride = 0;
        int ovrPosAlias = 0;
        int ovrNegAlias = 0;
        int ptrnEvict = 0;
        int ctxEvict = 0;

    } llbpstats;

};


struct LLBPXConfig {
  TSCLConfig tsclConfig;

    // Context hash function parameters
    int T = 3;  // Type of history
    int W = 8;  // Number of branches
    int D = 4;  // Lookahead delay for prefetching
    int S = 1;  // Shift

#define LLBP_CONSTRAINED

#ifdef LLBP_CONSTRAINED
    // Size of pattern sets and CD
    int numPatterns = 16;
    int numContexts = 1024*14;

    // Associativity of pattern sets and CD
    int ctxAssoc = 7;
    int ptrnAssoc = 4;

    // Tag widths
    int TTWidth = 13;
    int CTWidth = 14;

    // PB config
    int pbSize = 64;
    int pbAssoc = 4;
#else
    int numContexts = 1000000;
    int numPatterns = 1000000;

    int ctxAssoc = numContexts;
    int ptrnAssoc = numPatterns;

    int TTWidth = 20;
    int CTWidth = 31;

    int pbSize = 1;
    int pbAssoc = pbSize;
#endif


    int CtrWidth = 3;
    int ReplCtrWidth = 16; // unused
    int CtxReplCtrWidth = 3;


    int adaptThreshold = 7;
    // int histLenThreshold = 17;
    int histLenThreshold = 22;

    int CitSize =  6*1024;
    int CitAssoc = 6;

    // int BCTWidth = 31;
    int BCTWidth = 16;

    bool simulateTiming = false;
    int accessDelay = 4;


  void print() const {
    printf("LLBP Config: NumPatterns=%i, NumContexts=%i, ctxAssoc=%i, ptrnAssoc=%i, "
           "CtrWidth=%i, ReplCtrWidth=%i, CtxReplCtrWidth=%i, pbSize=%i, "
           "TTWidth=%i, CTWidth=%i, adaptThreshold=%i, histLenThreshold=%i, "
           "CitSize=%i, CitAssoc=%i, BctxTTWidth=%i, "
           "simMispFlush=%i, accessDelay=%i, "
           "\n ",
           numPatterns, numContexts, ctxAssoc, ptrnAssoc,
           CtrWidth, ReplCtrWidth, CtxReplCtrWidth, pbSize,
           TTWidth, CTWidth, adaptThreshold, histLenThreshold,
           CitSize, CitAssoc, BCTWidth,
           simulateTiming, accessDelay
           );
  }
};




/////////////////////////
// LLBP Predictor

// The LLBP predictor without simulating the prefetch latency.
class LLBPXTSCL64k : public LLBPX {
   public:
    LLBPXTSCL64k(void)
        : LLBPX(LLBPXConfig
            {
                .tsclConfig = TSCLConfig
                {
                    .tageConfig = Tage64kConfig,
                    .useSC = true,
                    .useLoop = true
                },
                .simulateTiming = false,
            })
    {}
};


// The LLBP predictor with simulating the prefetch latency.
class LLBPXTSCL64kTiming : public LLBPX {
   public:
    LLBPXTSCL64kTiming(void)
        : LLBPX(LLBPXConfig
            {
                .tsclConfig = TSCLConfig
                {
                    .tageConfig = Tage64kConfig,
                    .useSC = true,
                    .useLoop = true
                },
                .simulateTiming = true,
            })
    {}
};


};  // namespace LLBP
