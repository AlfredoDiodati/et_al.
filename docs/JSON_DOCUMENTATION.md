# json.h - JSON value tree (parse, build, write)

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose serialization utility, not tied to any model.

`json.h` implements a minimal JSON value tree: parse text/a file into it, build one programmatically, write it back out. It lives at the repository root as a standalone file (matching `ad.h`'s "core, general-purpose tool" placement), not under `frame/`, because its actual use case is different from `frame/`'s loaders: saving/loading **parameters** (hyperparameters, fit diagnostics, config), not bulk tabular data. A `DataFrame` never appears in this file's API and this file never appears in `frame/frame.h`'s.

Has zero dependency on `linalg/mat.h`/`mreal` — a parameter is typically a standalone scalar (a learning rate, an epoch count, a layer width), not part of a `Mat` computation, so `JSON_NUMBER` stores a plain `double` regardless of `-DMAT_DOUBLE`.

## Design: a flat tagged struct, not a union

```c
typedef enum { JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT } JsonType;

typedef struct JsonValue {
    JsonType type;
    int boolean;
    double number;
    char *string;
    struct JsonValue **items; int n_items;
    char **keys; struct JsonValue **values; int n_members;
} JsonValue;
```

Every `JsonValue` carries every field, most unused depending on `.type`, rather than a C `union` of the heterogeneous owned-pointer cases. JSON documents in this project's actual use case (parameter files) are small, so the wasted memory is irrelevant, and a flat struct is far simpler to get right than a union whose members have different ownership/free semantics.

## API reference

```c
/* construction */
JsonValue *json_null(void);
JsonValue *json_bool(int b);
JsonValue *json_number(double n);
JsonValue *json_string(const char *s);
JsonValue *json_array(void);
void json_array_push(JsonValue *arr, JsonValue *item);
JsonValue *json_object(void);
void json_object_set(JsonValue *obj, const char *key, JsonValue *value);
void json_free(JsonValue *v);

/* accessors (assert on type mismatch) */
JsonValue *json_object_get(const JsonValue *obj, const char *key); /* NULL if absent */
double json_as_number(const JsonValue *v);
const char *json_as_string(const JsonValue *v);
int json_as_bool(const JsonValue *v);
int json_array_len(const JsonValue *v);
JsonValue *json_array_get(const JsonValue *v, int i);

/* parse / write */
JsonValue *json_parse(const char *text);
JsonValue *json_parse_file(const char *path);
char *json_write(const JsonValue *v);            /* malloc'd string, caller frees */
void json_write_file(const JsonValue *v, const char *path);
```

`json_array_push`/`json_object_set` take **ownership** of the value pointer passed in (no copy) — build a value with `json_number`/`json_string`/etc. specifically to hand it to a container. `json_object_set`'s **key** is copied instead, since callers typically pass a string literal or a key they keep using elsewhere. Setting a key that already exists frees the old value and replaces it — last value for a key wins, matching standard JSON semantics (and this file's own parser goes through the same function, so a duplicate key in a parsed document behaves identically).

```c
#include <json.h>

JsonValue *cfg = json_object();
json_object_set(cfg, "lr", json_number(0.05));
JsonValue *sizes = json_array();
json_array_push(sizes, json_number(2));
json_array_push(sizes, json_number(4));
json_array_push(sizes, json_number(1));
json_object_set(cfg, "sizes", sizes);
json_write_file(cfg, "run_config.json");
json_free(cfg);

JsonValue *loaded = json_parse_file("run_config.json");
double lr = json_as_number(json_object_get(loaded, "lr"));
json_free(loaded);
```

## Scope and known simplifications

- `\uXXXX` escapes decode to UTF-8 for BMP code points only — no surrogate-pair handling for characters outside it. Sufficient for parameter-file text; not a full Unicode-in-JSON implementation.
- Number parsing uses `strtod`, which is more lenient than strict JSON grammar (accepts a leading `+`, leading zeros, hex floats, ...). Not a hidden bug: this file's own writer never produces anything `strtod` would accept but JSON wouldn't, so round-trip correctness is unaffected — only reading slightly-malformed *external* JSON differs from a strict-grammar parser.
- `json_write` is compact (no indentation) — this is for machine round-tripping, not human-edited config files. Verified to interoperate correctly with Python's `json` module in both directions during development (write here → `json.load` in Python; `json.dump` in Python, pretty-printed with `\uXXXX` escapes → parse here) — not shipped as part of the test suite (which stays pure C, no Python dependency at test time), but confirms this isn't just internally self-consistent.
- Malformed input (bad escape, mismatched brackets, trailing garbage) is a contract violation (`assert`), not a recoverable parse error — the same convention `linalg/decomp.h`/`linalg/solver.h` and `frame/`'s loaders already use.

## Memory ownership

Every `JsonValue` owns everything reachable from it (strings, array items, object keys/values) — `json_free` recursively frees the whole tree. `json_parse`/`json_parse_file` return a tree the caller must `json_free()`. `json_write`/nothing here aliases caller memory except the `value`/`item` pointers explicitly handed to `json_array_push`/`json_object_set`, which take ownership as documented above.

## Testing

`tests/correctness/test_json.c` uses **round-trip structural equality** (`json_deep_equal`) as its correctness oracle throughout — parse, or build, a tree; write it; parse the output; assert the result is deep-equal to the original. This is the strongest correctness technique available for a parser+writer pair with no independent reference implementation in C to compare against (unlike, say, a naive triple-loop matmul reference): if the two sides agree with each other across many varied and adversarial inputs, both are almost certainly right. `json_deep_equal` compares `JSON_NUMBER` with exact `==`, not a tolerance — deliberately: this is a text-round-trip-exactness check, not a numerical-computation comparison (where this project always uses a tolerance), and `json_write`'s `%.17g` is `double`'s shortest-round-trip digit count, so exact equality is the *correct* check here, not a violation of "never compare floats with `==`."

Fixed cases cover: a known hand-written document (nested object/array/bool/null), string escaping (quotes, backslash, newline) round-tripped, a `\uXXXX` Unicode escape decoded to UTF-8 and verified byte-for-byte, the construction API round-tripped, duplicate-key last-value-wins, adversarial shapes (empty object/array, 5-deep nested arrays, negative/zero/exponent-notation numbers), and a real file round-trip (`json_write_file`/`json_parse_file`).

Under `STRESS=1`, `test_random_roundtrip_stress` generates 200 random JSON trees (fixed seed `42`, depth capped at 4) and round-trips each — per this project's testing policy of fixing the seed for reproducibility while biasing generation toward fragile regions rather than uniform noise: numbers deliberately include zero, negative, fractional, and very large/small (`1e300`/`1e-300`) magnitudes; strings are built from a pool weighted toward characters that need escaping (`"`, `\`, newline, tab, CR, and raw control bytes `\x01`/`\x1f`) rather than plain alphanumeric text. Verified clean under ASan/UBSan (default and stress) given how much manual `malloc`/`realloc`/`free` this file's tokenizer and tree-builder do.

## Known limitations and future work

- No streaming parser — the whole document is read into memory first (`json_parse_file` via a whole-file read), same tradeoff `frame/csv.h`/`frame/txt.h` make.
- No pretty-printing option on the writer — always compact.
- No `frame/`-side `DataFrame`↔JSON loader (e.g. an array of objects as rows) — this file is scoped to parameters, not bulk tabular data; that would be a separate, `frame/`-tier concern if it's ever needed.
