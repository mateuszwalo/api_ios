#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Boundary between Swift and llama.cpp.
///
/// Everything C++ lives behind this header: the model, the context, `mtmd` for vision,
/// and `json_schema_to_grammar` from llama.cpp's `common/`. Swift sees Foundation types
/// only, so the app target never needs the C++ toolchain.
///
/// Three llama.cpp facilities are reused deliberately rather than reimplemented:
///
///   * `json_schema_to_grammar` — already handles `$defs`, `$ref`, `anyOf`, `enum`,
///     `minItems`, `maxLength` and `minimum`, which is exactly the set that arrives in
///     practice.
///   * the chat template, read from the GGUF's own metadata. A hand-written template
///     would drift from the one the reference runtime applies, and the divergence would
///     look like a hardware difference rather than a bug.
///   * `mtmd` for images, the only route to a multimodal projector on iOS.

#pragma mark - Configuration

/// Load-time parameters. Defaults match the reference configuration; changing any of them
/// invalidates comparison with measurements taken elsewhere. See docs/DECISIONS.md.
@interface LLMLoadOptions : NSObject
@property (nonatomic) NSInteger contextLength;      // default 32768
@property (nonatomic) NSInteger threadCount;        // 0 = derive from active processors
@property (nonatomic) NSInteger batchSize;          // n_batch
@property (nonatomic) NSInteger microBatchSize;     // n_ubatch
@property (nonatomic) BOOL flashAttention;          // default YES
@property (nonatomic) BOOL useMemoryMapping;        // default YES
/// When NO (the default), the KV cache is cleared between requests so every prefill is
/// measured cold. Leaving it on is realistic for deployment but hides the true prefill
/// cost and makes a run depend on whatever happened to run before it.
@property (nonatomic) BOOL reuseKVCacheBetweenRequests;
+ (instancetype)defaults;
@end

#pragma mark - Prompt

@interface LLMImage : NSObject
@property (nonatomic, readonly) NSData *data;
@property (nonatomic, readonly) NSString *mimeType;
- (instancetype)initWithData:(NSData *)data mimeType:(NSString *)mimeType;
@end

@interface LLMTurn : NSObject
@property (nonatomic, readonly) NSString *role;
@property (nonatomic, readonly) NSString *text;
@property (nonatomic, readonly) NSArray<LLMImage *> *images;
- (instancetype)initWithRole:(NSString *)role
                        text:(NSString *)text
                      images:(NSArray<LLMImage *> *)images;
@end

#pragma mark - Generation

@interface LLMGenerationOptions : NSObject
@property (nonatomic) NSInteger maxTokens;
@property (nonatomic) double temperature;
/// GBNF source, or nil for unconstrained generation.
@property (nonatomic, copy, nullable) NSString *grammar;
@property (nonatomic, copy) NSArray<NSString *> *stopSequences;
/// Negative means unset; the sampler then uses its own default seeding.
@property (nonatomic) int64_t seed;
+ (instancetype)defaults;
@end

@interface LLMGenerationResult : NSObject
@property (nonatomic, readonly) NSString *text;
@property (nonatomic, readonly) NSInteger promptTokens;     // includes image tokens
@property (nonatomic, readonly) NSInteger completionTokens;
@property (nonatomic, readonly) NSInteger prefillMilliseconds;
@property (nonatomic, readonly) NSInteger decodeMilliseconds;
/// YES when the token budget ran out, NO when the model emitted a stop token.
@property (nonatomic, readonly) BOOL hitTokenLimit;
@end

#pragma mark - Bridge

extern NSErrorDomain const LLMBridgeErrorDomain;

/// A plain NS_ENUM rather than NS_ERROR_ENUM, so the Swift name is predictable.
///
/// NS_ERROR_ENUM does not import under the name written here: it produces a wrapper type
/// with the codes nested inside it, and Swift could not find `LLMBridgeErrorCode` at all.
/// With NS_ENUM the type keeps its name and the cases lose the shared prefix, which is the
/// spelling the Swift side already expects.
typedef NS_ENUM(NSInteger, LLMBridgeErrorCode) {
    LLMBridgeErrorModelLoadFailed = 1,
    LLMBridgeErrorProjectorLoadFailed,
    LLMBridgeErrorNotLoaded,
    LLMBridgeErrorTokenizeFailed,
    LLMBridgeErrorContextOverflow,
    LLMBridgeErrorGrammarInvalid,
    LLMBridgeErrorImageRejected,
    LLMBridgeErrorDecodeFailed,
    LLMBridgeErrorCancelled,
};

@interface LLMBridge : NSObject

/// Compiles a JSON Schema into GBNF.
///
/// A pure function needing no model, which is what lets the conversion be tested in CI on
/// a simulator without a device or a 3 GB file. That matters, because this is the
/// component most likely to be wrong and least likely to announce it: a grammar that is
/// merely too permissive yields plausible output that no longer matches the schema.
///
/// `schemaJSON` is the schema as it arrived on the wire, including
/// `"additionalProperties": false`, which the client library adds to every object.
/// Returns nil and populates `error` when the schema uses an unsupported construct.
/// Callers must surface that as a failure and never fall back to describing the schema in
/// the prompt.
+ (nullable NSString *)grammarFromJSONSchema:(NSString *)schemaJSON
                                       error:(NSError **)error;

- (BOOL)loadModelAtPath:(NSString *)modelPath
          projectorPath:(nullable NSString *)projectorPath
                options:(LLMLoadOptions *)options
                  error:(NSError **)error;

- (void)unload;

@property (nonatomic, readonly, getter=isLoaded) BOOL loaded;
@property (nonatomic, readonly, nullable) NSString *loadedModelPath;
@property (nonatomic, readonly) NSInteger contextLength;
/// YES when a multimodal projector is loaded. Images are rejected outright otherwise,
/// rather than dropped — a vision request answered from the text alone looks like a
/// quality problem and is in fact a configuration one.
@property (nonatomic, readonly) BOOL supportsImages;

/// Token count for a prompt, images included, without generating anything.
/// Returns -1 on failure and populates `error`.
- (NSInteger)measurePromptTokens:(NSArray<LLMTurn *> *)turns
                           error:(NSError **)error;

/// The prompt exactly as the chat template renders it, for inspection.
///
/// Exposed because a template mismatch is invisible from the outside: the same weights with
/// a different turn marker produce different output, and the difference looks like a
/// hardware or quantisation effect. Seeing the rendered text settles it in one glance.
- (nullable NSString *)renderedPromptForTurns:(NSArray<LLMTurn *> *)turns;

/// Runs one generation to completion. Blocking: callers serialise it and keep it off the
/// main thread. `isCancelled` is polled between tokens so a departed client stops costing
/// time.
- (nullable LLMGenerationResult *)generateWithTurns:(NSArray<LLMTurn *> *)turns
                                            options:(LLMGenerationOptions *)options
                                        isCancelled:(BOOL (^_Nullable)(void))isCancelled
                                              error:(NSError **)error;

/// The last lines llama.cpp itself printed.
///
/// Its diagnostics go to stderr, which on a sideloaded device nobody can read. When loading
/// fails, the reason is almost always in there — an unsupported kernel, a backend that would
/// not initialise, an allocation that was refused and by how much — and without it the
/// caller is left inferring from a null return. Kept so a failure can be read over HTTP.
+ (NSArray<NSString *> *)recentEngineLog;

/// Records an application event in the same file as llama.cpp's own output.
///
/// A crash leaves the last lines standing; interleaving the app's own milestones with the
/// engine's — model loading with these settings, request with a grammar started — is what
/// turns "it died" into "it died on the first token of a schema-constrained request".
+ (void)noteEvent:(NSString *)event;

/// Bytes the OS attributes to this process, from `phys_footprint`.
///
/// Not RSS: the memory killer accounts for footprint, so that is the figure a memory
/// budget must be judged against.
+ (uint64_t)physicalFootprintBytes;

/// Bytes still available before this process reaches its limit.
+ (uint64_t)availableMemoryBytes;

@end

NS_ASSUME_NONNULL_END
