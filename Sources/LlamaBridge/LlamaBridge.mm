#import "LlamaBridge.h"

#import <mach/mach.h>
#import <os/proc.h>
#import <time.h>

#include <mutex>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
// Pulls in common_json. llama.cpp replaced nlohmann in this interface: the converter now
// takes the project's own JSON type, so there is no nlohmann include here any more.
#include "json-schema-to-grammar.h"

NSErrorDomain const LLMBridgeErrorDomain = @"LLMBridgeErrorDomain";

#pragma mark - Engine log

// llama.cpp reports why it failed through its log callback and nowhere else. On a sideloaded
// device stderr goes nowhere anybody can read, so the last lines are kept here and served
// over HTTP. Without them a refused context is just a null return and a guess.
static std::mutex gLogMutex;
static std::vector<std::string> gLogLines;
static const size_t kLogLineLimit = 300;
static FILE *gLogFile = nullptr;

// Written to disk as well as kept in memory, and flushed line by line.
//
// The failures worth reading are the ones that end the process: mtmd aborts on a mismatched
// projector rather than returning an error, and the in-memory copy dies with the app. A file
// flushed after every line survives that, so the next launch can serve what the last one
// said on its way out. llama.cpp logs rarely outside of loading, so the cost does not show.
static NSString *EngineLogPath(void) {
    NSString *documents = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,
                                                              NSUserDomainMask, YES).firstObject;
    NSString *directory = [documents stringByAppendingPathComponent:@"logs"];
    [[NSFileManager defaultManager] createDirectoryAtPath:directory
                              withIntermediateDirectories:YES attributes:nil error:nil];
    return [directory stringByAppendingPathComponent:@"engine.log"];
}

static void OpenLogFileLocked(void) {
    if (gLogFile != nullptr) return;
    NSString *path = EngineLogPath();
    // Start fresh past a couple of megabytes: the tail is what gets read, and an unbounded
    // file is a poor neighbour on a device being filled with 3 GB models.
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil];
    if ([attributes fileSize] > 2 * 1024 * 1024) {
        [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
    }
    gLogFile = fopen(path.fileSystemRepresentation, "a");
}

static void AppendLogLine(const char *text) {
    if (text == nullptr) return;
    std::string line(text);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (line.empty()) return;

    std::lock_guard<std::mutex> lock(gLogMutex);
    gLogLines.push_back(line);
    if (gLogLines.size() > kLogLineLimit) {
        gLogLines.erase(gLogLines.begin(), gLogLines.begin() + (gLogLines.size() - kLogLineLimit));
    }
    OpenLogFileLocked();
    if (gLogFile != nullptr) {
        fputs(line.c_str(), gLogFile);
        fputc('\n', gLogFile);
        fflush(gLogFile);
    }
}

static NSError *MakeError(LLMBridgeErrorCode code, NSString *message) {
    return [NSError errorWithDomain:LLMBridgeErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey: message ?: @"unknown error"}];
}

#pragma mark - Pinned API notes

// This file is written against the exact llama.cpp tag in LLAMA_CPP_TAG, and the few places
// where that matters are called out where they appear. An earlier version tried to detect
// the API shape at compile time with `if constexpr`; that was the wrong instinct twice over.
// In a non-template function `if constexpr` discards nothing, so the dead branch still has
// to compile — which is exactly how it failed — and the hedging hid which API was actually
// in use. A pinned dependency should be read, not guessed at.

#pragma mark - Value types

@implementation LLMLoadOptions
+ (instancetype)defaults {
    LLMLoadOptions *o = [LLMLoadOptions new];
    o.contextLength = 32768;
    o.threadCount = 0;
    o.batchSize = 512;
    o.microBatchSize = 512;
    o.flashAttention = YES;
    o.useMemoryMapping = YES;
    o.reuseKVCacheBetweenRequests = NO;
    return o;
}
@end

@implementation LLMImage
- (instancetype)initWithData:(NSData *)data mimeType:(NSString *)mimeType {
    if ((self = [super init])) {
        _data = data;
        _mimeType = [mimeType copy];
    }
    return self;
}
@end

@implementation LLMTurn
- (instancetype)initWithRole:(NSString *)role
                        text:(NSString *)text
                      images:(NSArray<LLMImage *> *)images {
    if ((self = [super init])) {
        _role = [role copy];
        _text = [text copy];
        _images = [images copy] ?: @[];
    }
    return self;
}
@end

@implementation LLMGenerationOptions
+ (instancetype)defaults {
    LLMGenerationOptions *o = [LLMGenerationOptions new];
    o.maxTokens = 4096;
    o.temperature = 0.0;
    o.grammar = nil;
    o.stopSequences = @[];
    o.seed = -1;
    return o;
}
@end

@interface LLMGenerationResult ()
@property (nonatomic, readwrite) NSString *text;
@property (nonatomic, readwrite) NSInteger promptTokens;
@property (nonatomic, readwrite) NSInteger completionTokens;
@property (nonatomic, readwrite) NSInteger prefillMilliseconds;
@property (nonatomic, readwrite) NSInteger decodeMilliseconds;
@property (nonatomic, readwrite) BOOL hitTokenLimit;
@end

@implementation LLMGenerationResult
@end

#pragma mark - Bridge

@implementation LLMBridge {
    llama_model *_model;
    llama_context *_ctx;
    mtmd_context *_mtmd;
    LLMLoadOptions *_options;
    NSString *_modelPath;
    NSLock *_lock;
}

- (instancetype)init {
    if ((self = [super init])) {
        _lock = [NSLock new];
        _options = [LLMLoadOptions defaults];
        static dispatch_once_t once;
        dispatch_once(&once, ^{
            // Marks each launch, so lines from consecutive processes — including one that
            // crashed — can be told apart when the file is read back.
            AppendLogLine("===== engine started =====");

            // Assertion failures bypass the log callback entirely: GGML_ASSERT goes straight
            // to ggml_abort, which prints and kills the process. That is the one message that
            // explains a crash, and it was the one never written down. The abort callback
            // sees it first, so it goes to the log file — flushed — before the process ends.
            ggml_set_abort_callback([](const char *message) {
                std::string line = std::string("FATAL: ") + (message ? message : "(no message)");
                AppendLogLine(line.c_str());
                if (message) fputs(message, stderr);
            });

            llama_backend_init();
            // llama.cpp is chatty at info level and every line costs time on a device that
            // is being timed. Warnings and errors still come through.
            llama_log_set([](enum ggml_log_level level, const char *text, void *) {
                AppendLogLine(text);
                if (level >= GGML_LOG_LEVEL_WARN && text != nullptr) fputs(text, stderr);
            }, nullptr);
        });
    }
    return self;
}

- (void)dealloc {
    [self unload];
}

#pragma mark Grammar

+ (nullable NSString *)grammarFromJSONSchema:(NSString *)schemaJSON error:(NSError **)error {
    if (schemaJSON.length == 0) {
        if (error) *error = MakeError(LLMBridgeErrorGrammarInvalid, @"empty schema");
        return nil;
    }
    try {
        // common_json, not nlohmann: llama.cpp carries its own JSON type now, and
        // json_schema_to_grammar takes that. Its parser preserves property order, which
        // matters — the converter emits object rules in the order the properties appear,
        // and reordering them changes which outputs the grammar accepts.
        common_json schema = common_json::parse(std::string(schemaJSON.UTF8String));
        std::string gbnf = json_schema_to_grammar(schema);
        if (gbnf.empty()) {
            if (error) *error = MakeError(LLMBridgeErrorGrammarInvalid,
                                          @"the converter produced an empty grammar");
            return nil;
        }
        return [NSString stringWithUTF8String:gbnf.c_str()];
    } catch (const std::exception &e) {
        if (error) *error = MakeError(LLMBridgeErrorGrammarInvalid,
                                      [NSString stringWithUTF8String:e.what()]);
        return nil;
    } catch (...) {
        if (error) *error = MakeError(LLMBridgeErrorGrammarInvalid,
                                      @"unsupported JSON Schema construct");
        return nil;
    }
}

#pragma mark Lifecycle

- (BOOL)loadModelAtPath:(NSString *)modelPath
          projectorPath:(nullable NSString *)projectorPath
                options:(LLMLoadOptions *)options
                  error:(NSError **)error {
    [_lock lock];
    @try {
        [self unloadLocked];
        _options = options ?: [LLMLoadOptions defaults];

        llama_model_params mparams = llama_model_default_params();
        // Everything on the GPU. A partially offloaded model on Apple silicon is slower
        // than either extreme, and unified memory makes the split pointless anyway.
        mparams.n_gpu_layers = 999;
        // Memory mapping is selected through load_mode; the old use_mmap flag is gone.
        mparams.load_mode = _options.useMemoryMapping ? LLAMA_LOAD_MODE_MMAP
                                                      : LLAMA_LOAD_MODE_NONE;

        _model = llama_model_load_from_file(modelPath.fileSystemRepresentation, mparams);
        if (_model == nullptr) {
            if (error) *error = MakeError(LLMBridgeErrorModelLoadFailed,
                                          [NSString stringWithFormat:@"could not load %@",
                                           modelPath.lastPathComponent]);
            return NO;
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t)_options.contextLength;
        cparams.n_batch = (uint32_t)MAX(_options.batchSize, 1);
        cparams.n_ubatch = (uint32_t)MAX(_options.microBatchSize, 1);
        cparams.n_threads = (int32_t)(_options.threadCount > 0
                                      ? _options.threadCount
                                      : NSProcessInfo.processInfo.activeProcessorCount);
        cparams.n_threads_batch = cparams.n_threads;

        // One set of logits, not one per token in the batch.
        //
        // The output buffer is sized n_vocab × n_outputs, and Gemma 3 carries a 262144-token
        // vocabulary: at the default, where n_outputs follows n_batch, that is half a
        // gigabyte reserved before a single token is produced. This server samples one token
        // at a time and never reads logits for any position but the last, so the rest was
        // never going to be looked at. On an 8 GB device it is the difference between the
        // context being created and the load failing outright.
        cparams.n_outputs_max = 1;
        cparams.n_outputs_max_per_seq = 1;
        cparams.n_seq_max = 1;
        // AUTO, not ENABLED, when it is wanted.
        //
        // Forcing it makes context creation fail outright wherever the backend has no kernel
        // for the model's head size — Gemma 3 uses 256, which is not universally covered —
        // and the failure arrives as a null context with no explanation attached. AUTO lets
        // llama.cpp use flash attention where it can and quietly do without where it cannot,
        // which is the behaviour wanted here: a slower server beats one that will not start.
        cparams.flash_attn_type = _options.flashAttention ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                                          : LLAMA_FLASH_ATTN_TYPE_DISABLED;

        _ctx = llama_init_from_model(_model, cparams);
        if (_ctx == nullptr) {
            llama_model_free(_model);
            _model = nullptr;
            if (error) *error = MakeError(LLMBridgeErrorModelLoadFailed,
                                          @"could not create a context (out of memory?)");
            return NO;
        }

        if (projectorPath.length > 0) {
            mtmd_context_params vparams = mtmd_context_params_default();
            vparams.use_gpu = true;
            vparams.print_timings = false;
            vparams.n_threads = cparams.n_threads;
            vparams.media_marker = mtmd_default_marker();
            _mtmd = mtmd_init_from_file(projectorPath.fileSystemRepresentation, _model, vparams);
            if (_mtmd == nullptr) {
                [self unloadLocked];
                if (error) *error = MakeError(LLMBridgeErrorProjectorLoadFailed,
                                              @"could not load the multimodal projector");
                return NO;
            }
        }

        _modelPath = [modelPath copy];
        return YES;
    } @finally {
        [_lock unlock];
    }
}

- (void)unload {
    [_lock lock];
    [self unloadLocked];
    [_lock unlock];
}

- (void)unloadLocked {
    if (_mtmd) { mtmd_free(_mtmd); _mtmd = nullptr; }
    if (_ctx) { llama_free(_ctx); _ctx = nullptr; }
    if (_model) { llama_model_free(_model); _model = nullptr; }
    _modelPath = nil;
}

- (BOOL)isLoaded { return _ctx != nullptr; }
- (nullable NSString *)loadedModelPath { return _modelPath; }
- (NSInteger)contextLength { return _ctx ? (NSInteger)llama_n_ctx(_ctx) : 0; }
- (BOOL)supportsImages { return _mtmd != nullptr && mtmd_support_vision(_mtmd); }

#pragma mark Prompt assembly

/// Renders the conversation with the chat template carried in the model file.
///
/// The template comes from the GGUF metadata rather than from this file on purpose: a
/// hand-written Gemma template differing by one token from the one the reference runtime
/// applies produces different output from identical weights, and that difference would be
/// read as a hardware effect rather than as the bug it is.
- (std::string)renderPrompt:(NSArray<LLMTurn *> *)turns hasImages:(BOOL *)outHasImages {
    std::vector<std::string> roles, contents;
    roles.reserve(turns.count);
    contents.reserve(turns.count);
    BOOL any = NO;

    const char *marker = mtmd_default_marker();
    for (LLMTurn *turn in turns) {
        std::string content;
        // The media marker precedes the text: that is the part order the reference client
        // sends, and the order the model was shown them in when the reference numbers
        // were taken.
        for (NSUInteger i = 0; i < turn.images.count; i++) {
            content += marker;
            content += "\n";
            any = YES;
        }
        content += turn.text.UTF8String ? turn.text.UTF8String : "";
        roles.push_back(turn.role.UTF8String ? turn.role.UTF8String : "user");
        contents.push_back(content);
    }
    if (outHasImages) *outHasImages = any;

    std::vector<llama_chat_message> messages;
    messages.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); i++) {
        messages.push_back({roles[i].c_str(), contents[i].c_str()});
    }

    const char *tmpl = llama_model_chat_template(_model, nullptr);
    std::vector<char> buf(8192);
    int32_t needed = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                               /*add_ass=*/true, buf.data(), (int32_t)buf.size());
    if (needed > (int32_t)buf.size()) {
        buf.resize(needed + 1);
        needed = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                           true, buf.data(), (int32_t)buf.size());
    }
    if (needed < 0) return std::string();
    return std::string(buf.data(), needed);
}

- (std::vector<llama_token>)tokenizeText:(const std::string &)text {
    const llama_vocab *vocab = llama_model_get_vocab(_model);
    int32_t upper = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                    nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
    std::vector<llama_token> tokens(upper > 0 ? upper : 0);
    if (upper > 0) {
        llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                       tokens.data(), (int32_t)tokens.size(), true, true);
    }
    return tokens;
}

#pragma mark Token measurement

- (NSInteger)measurePromptTokens:(NSArray<LLMTurn *> *)turns error:(NSError **)error {
    [_lock lock];
    @try {
        if (!_ctx) {
            if (error) *error = MakeError(LLMBridgeErrorNotLoaded, @"no model is loaded");
            return -1;
        }
        BOOL hasImages = NO;
        std::string prompt = [self renderPrompt:turns hasImages:&hasImages];
        if (prompt.empty()) {
            if (error) *error = MakeError(LLMBridgeErrorTokenizeFailed,
                                          @"the chat template produced no prompt");
            return -1;
        }
        if (!hasImages) {
            return (NSInteger)[self tokenizeText:prompt].size();
        }
        if (!_mtmd) {
            if (error) *error = MakeError(LLMBridgeErrorImageRejected,
                                          @"the request carries an image but no projector is loaded");
            return -1;
        }
        // mtmd_tokenize only splits and counts; the vision encoder runs later, during
        // evaluation. Measuring is therefore cheap even for a full page image.
        mtmd_input_chunks *chunks = nullptr;
        NSInteger count = [self tokenizeMultimodal:turns prompt:prompt chunks:&chunks error:error];
        if (chunks) mtmd_input_chunks_free(chunks);
        return count;
    } @finally {
        [_lock unlock];
    }
}

- (nullable NSString *)renderedPromptForTurns:(NSArray<LLMTurn *> *)turns {
    [_lock lock];
    @try {
        if (!_model) return nil;
        BOOL hasImages = NO;
        std::string prompt = [self renderPrompt:turns hasImages:&hasImages];
        if (prompt.empty()) return nil;
        return [NSString stringWithUTF8String:prompt.c_str()];
    } @finally {
        [_lock unlock];
    }
}

- (NSInteger)tokenizeMultimodal:(NSArray<LLMTurn *> *)turns
                         prompt:(const std::string &)prompt
                         chunks:(mtmd_input_chunks **)outChunks
                          error:(NSError **)error {
    // The helper returns a wrapper rather than a bitmap: it can also open a video, in which
    // case it hands back a decoder context that owns the frames. Images never produce one,
    // and this server rejects anything but an image, but the context is released anyway so
    // the invariant does not depend on that staying true.
    std::vector<mtmd_bitmap *> bitmaps;
    const mtmd_helper_init_opt bitmapOptions = mtmd_helper_init_opt_default();
    for (LLMTurn *turn in turns) {
        for (LLMImage *image in turn.images) {
            mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_buf(
                _mtmd, (const unsigned char *)image.data.bytes, image.data.length,
                /*placeholder=*/false, bitmapOptions);
            if (wrapper.video_ctx != nullptr) mtmd_helper_video_free(wrapper.video_ctx);
            if (wrapper.bitmap == nullptr) {
                for (auto *b : bitmaps) mtmd_bitmap_free(b);
                if (error) *error = MakeError(LLMBridgeErrorImageRejected,
                                              @"the image could not be decoded");
                return -1;
            }
            bitmaps.push_back(wrapper.bitmap);
        }
    }

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    mtmd_input_text text{};
    text.text = prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    int32_t rc = mtmd_tokenize(_mtmd, chunks, &text,
                               (const mtmd_bitmap **)bitmaps.data(), bitmaps.size());
    for (auto *b : bitmaps) mtmd_bitmap_free(b);

    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        if (error) *error = MakeError(LLMBridgeErrorTokenizeFailed,
                                      [NSString stringWithFormat:
                                       @"multimodal tokenization failed (%d); the image count and "
                                       @"the number of markers in the prompt must agree", rc]);
        return -1;
    }

    NSInteger total = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        total += (NSInteger)mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks, i));
    }
    if (outChunks) { *outChunks = chunks; } else { mtmd_input_chunks_free(chunks); }
    return total;
}

#pragma mark Generation

- (nullable LLMGenerationResult *)generateWithTurns:(NSArray<LLMTurn *> *)turns
                                            options:(LLMGenerationOptions *)options
                                        isCancelled:(BOOL (^_Nullable)(void))isCancelled
                                              error:(NSError **)error {
    [_lock lock];
    @try {
        if (!_ctx || !_model) {
            if (error) *error = MakeError(LLMBridgeErrorNotLoaded, @"no model is loaded");
            return nil;
        }

        const llama_vocab *vocab = llama_model_get_vocab(_model);
        const uint64_t tStart = clock_gettime_nsec_np(CLOCK_MONOTONIC);

        // A cold cache unless reuse was asked for: otherwise the measured prefill depends
        // on whatever request happened to run before this one.
        if (!_options.reuseKVCacheBetweenRequests) {
            llama_memory_clear(llama_get_memory(_ctx), true);
        }

        BOOL hasImages = NO;
        std::string prompt = [self renderPrompt:turns hasImages:&hasImages];
        if (prompt.empty()) {
            if (error) *error = MakeError(LLMBridgeErrorTokenizeFailed,
                                          @"the chat template produced no prompt");
            return nil;
        }
        if (hasImages && !_mtmd) {
            if (error) *error = MakeError(LLMBridgeErrorImageRejected,
                                          @"the request carries an image but no projector is loaded");
            return nil;
        }

        const int32_t nCtx = (int32_t)llama_n_ctx(_ctx);
        NSInteger promptTokens = 0;
        llama_pos nPast = 0;

        // ---- prefill ----
        if (hasImages) {
            mtmd_input_chunks *chunks = nullptr;
            promptTokens = [self tokenizeMultimodal:turns prompt:prompt chunks:&chunks error:error];
            if (promptTokens < 0) return nil;
            if (promptTokens >= nCtx) {
                mtmd_input_chunks_free(chunks);
                if (error) *error = MakeError(LLMBridgeErrorContextOverflow,
                                              [NSString stringWithFormat:
                                               @"prompt of %ld tokens does not fit in a context of %d",
                                               (long)promptTokens, nCtx]);
                return nil;
            }
            llama_pos newPast = 0;
            int32_t rc = mtmd_helper_eval_chunks(_mtmd, _ctx, chunks, /*n_past=*/0, /*seq_id=*/0,
                                                 (int32_t)MAX(_options.batchSize, 1),
                                                 /*logits_last=*/true, &newPast);
            mtmd_input_chunks_free(chunks);
            if (rc != 0) {
                if (error) *error = MakeError(LLMBridgeErrorDecodeFailed,
                                              @"the vision encoder or the prefill failed");
                return nil;
            }
            nPast = newPast;
        } else {
            std::vector<llama_token> tokens = [self tokenizeText:prompt];
            promptTokens = (NSInteger)tokens.size();
            if (promptTokens == 0) {
                if (error) *error = MakeError(LLMBridgeErrorTokenizeFailed, @"empty prompt");
                return nil;
            }
            if (promptTokens >= nCtx) {
                if (error) *error = MakeError(LLMBridgeErrorContextOverflow,
                                              [NSString stringWithFormat:
                                               @"prompt of %ld tokens does not fit in a context of %d",
                                               (long)promptTokens, nCtx]);
                return nil;
            }
            const int32_t step = (int32_t)MAX(_options.batchSize, 1);
            for (int32_t i = 0; i < (int32_t)tokens.size(); i += step) {
                if (isCancelled && isCancelled()) {
                    if (error) *error = MakeError(LLMBridgeErrorCancelled, @"cancelled");
                    return nil;
                }
                const int32_t n = MIN(step, (int32_t)tokens.size() - i);
                llama_batch batch = llama_batch_get_one(tokens.data() + i, n);
                if (llama_decode(_ctx, batch) != 0) {
                    if (error) *error = MakeError(LLMBridgeErrorDecodeFailed, @"prefill failed");
                    return nil;
                }
                nPast += n;
            }
        }

        // ---- sampler ----
        llama_sampler *chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (options.grammar.length > 0) {
            // The grammar goes first: it masks the tokens the schema forbids before any
            // other sampler chooses among them. Placed after a selection step it could
            // only veto, which under greedy sampling means failing rather than picking the
            // best allowed token.
            llama_sampler *g = llama_sampler_init_grammar(vocab, options.grammar.UTF8String, "root");
            if (g == nullptr) {
                llama_sampler_free(chain);
                if (error) *error = MakeError(LLMBridgeErrorGrammarInvalid,
                                              @"the grammar was rejected by the sampler");
                return nil;
            }
            llama_sampler_chain_add(chain, g);
        }
        if (options.temperature > 0.0) {
            // Honouring an explicit temperature means not also forcing top_k 1, which
            // would make the setting a no-op. The reference client sends 0, so the
            // reference path is the greedy one below and is unaffected.
            llama_sampler_chain_add(chain, llama_sampler_init_temp((float)options.temperature));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(
                options.seed >= 0 ? (uint32_t)options.seed : LLAMA_DEFAULT_SEED));
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        }

        // ---- decode ----
        std::string out;
        NSInteger completion = 0;
        BOOL hitLimit = NO;
        uint64_t tFirstToken = 0;
        char piece[512];

        const NSInteger budget = MAX(options.maxTokens, 1);
        while (true) {
            if (isCancelled && isCancelled()) {
                llama_sampler_free(chain);
                if (error) *error = MakeError(LLMBridgeErrorCancelled, @"cancelled");
                return nil;
            }

            // llama_sampler_sample accepts the token itself. It must not be accepted again.
            //
            // An earlier version did, and for a plain greedy sampler that is harmless — accept
            // is a no-op there. For the grammar sampler it is not: the token is applied to the
            // grammar state twice, the state advances past where the output actually is, and
            // the next token finds no legal continuation. llama.cpp treats that as a fatal
            // error and aborts the process. It surfaced on the first request ever to carry a
            // schema, which is the request this whole server exists to serve.
            llama_token token = llama_sampler_sample(chain, _ctx, -1);
            if (tFirstToken == 0) tFirstToken = clock_gettime_nsec_np(CLOCK_MONOTONIC);
            if (llama_vocab_is_eog(vocab, token)) break;

            const int32_t n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
            if (n > 0) out.append(piece, n);
            completion++;

            BOOL stopped = NO;
            for (NSString *stop in options.stopSequences) {
                const char *s = stop.UTF8String;
                if (s == nullptr || *s == '\0') continue;
                size_t at = out.rfind(s);
                if (at != std::string::npos) {
                    out.erase(at);
                    stopped = YES;
                    break;
                }
            }
            if (stopped) break;

            if (completion >= budget || nPast + 1 >= nCtx) { hitLimit = YES; break; }

            llama_batch batch = llama_batch_get_one(&token, 1);
            if (llama_decode(_ctx, batch) != 0) {
                llama_sampler_free(chain);
                if (error) *error = MakeError(LLMBridgeErrorDecodeFailed, @"generation failed");
                return nil;
            }
            nPast++;
        }
        llama_sampler_free(chain);

        const uint64_t tEnd = clock_gettime_nsec_np(CLOCK_MONOTONIC);
        if (tFirstToken == 0) tFirstToken = tEnd;

        if (!_options.reuseKVCacheBetweenRequests) {
            llama_memory_clear(llama_get_memory(_ctx), true);
        }

        LLMGenerationResult *result = [LLMGenerationResult new];
        result.text = [NSString stringWithUTF8String:out.c_str()] ?: @"";
        result.promptTokens = promptTokens;
        result.completionTokens = completion;
        // Prefill covers everything up to the first token being available: tokenisation,
        // image encoding and the prompt passes. That is what the caller waits through, and
        // what the published prefill figure is defined against.
        result.prefillMilliseconds = (NSInteger)((tFirstToken - tStart) / 1000000ULL);
        result.decodeMilliseconds = (NSInteger)((tEnd - tFirstToken) / 1000000ULL);
        result.hitTokenLimit = hitLimit;
        return result;
    } @finally {
        [_lock unlock];
    }
}

#pragma mark Memory

+ (void)noteEvent:(NSString *)event {
    std::string line = std::string("APP: ") + (event.UTF8String ? event.UTF8String : "");
    AppendLogLine(line.c_str());
}

+ (NSArray<NSString *> *)recentEngineLog {
    // The file first, not the in-memory copy: after a crash memory is empty, and the file
    // still holds what the previous process said before it died — the part worth reading.
    NSString *contents = [NSString stringWithContentsOfFile:EngineLogPath()
                                                   encoding:NSUTF8StringEncoding error:nil];
    if (contents.length > 0) {
        NSMutableArray<NSString *> *fromFile = [NSMutableArray array];
        for (NSString *line in [contents componentsSeparatedByCharactersInSet:
                                [NSCharacterSet newlineCharacterSet]]) {
            if (line.length > 0) [fromFile addObject:line];
        }
        NSUInteger start = fromFile.count > kLogLineLimit ? fromFile.count - kLogLineLimit : 0;
        return [fromFile subarrayWithRange:NSMakeRange(start, fromFile.count - start)];
    }
    std::lock_guard<std::mutex> lock(gLogMutex);
    NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithCapacity:gLogLines.size()];
    for (const auto &line : gLogLines) {
        NSString *converted = [NSString stringWithUTF8String:line.c_str()];
        if (converted) [lines addObject:converted];
    }
    return lines;
}

+ (uint64_t)physicalFootprintBytes {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.phys_footprint;
    }
    return 0;
}

+ (uint64_t)availableMemoryBytes {
    return os_proc_available_memory();
}

@end
