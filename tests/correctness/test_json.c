#include "../../json.h"

/* Deep structural equality, used as the round-trip oracle throughout this
   file: no independent reference JSON implementation exists in C to
   compare against (unlike e.g. a naive triple-loop matmul), so
   write-then-parse-then-compare-to-the-original is the strongest
   available correctness technique for a parser+writer pair - if they
   agree with each other on every random tree thrown at them, both are
   almost certainly right (see tests/correctness/test_json.c's STRESS
   block below). Uses exact == for JSON_NUMBER, not a tolerance - this is
   NOT a numerical-computation comparison (where this project always uses
   a tolerance), it is a text-round-trip-exactness check: json_write's
   %.17g is double's shortest-round-trip digit count, so any finite double
   must come back bit-identical, and a tolerance would hide a real
   round-trip bug instead of catching one. */
static int json_deep_equal(const JsonValue *a, const JsonValue *b) {
    if (a->type != b->type) return 0;
    switch (a->type) {
        case JSON_NULL:   return 1;
        case JSON_BOOL:   return a->boolean == b->boolean;
        case JSON_NUMBER: return a->number == b->number;
        case JSON_STRING: return strcmp(a->string, b->string) == 0;
        case JSON_ARRAY:
            if (a->n_items != b->n_items) return 0;
            for (int i = 0; i < a->n_items; i++)
                if (!json_deep_equal(a->items[i], b->items[i])) return 0;
            return 1;
        case JSON_OBJECT:
            if (a->n_members != b->n_members) return 0;
            for (int i = 0; i < a->n_members; i++) {
                JsonValue *bv = json_object_get(b, a->keys[i]);
                if (!bv || !json_deep_equal(a->values[i], bv)) return 0;
            }
            return 1;
    }
    return 0;
}

static void test_known_output_parse(void) {
    puts("known-output: parse a hand-written JSON document");

    const char *text = "{\"lr\": 0.05, \"sizes\": [2, 4, 1], \"tanh\": true, \"note\": null}";
    JsonValue *v = json_parse(text);
    assert(v->type == JSON_OBJECT);
    assert(json_as_number(json_object_get(v, "lr")) == 0.05);
    assert(json_as_bool(json_object_get(v, "tanh")) == 1);
    assert(json_object_get(v, "note")->type == JSON_NULL);

    JsonValue *sizes = json_object_get(v, "sizes");
    assert(json_array_len(sizes) == 3);
    assert(json_as_number(json_array_get(sizes, 0)) == 2);
    assert(json_as_number(json_array_get(sizes, 1)) == 4);
    assert(json_as_number(json_array_get(sizes, 2)) == 1);

    json_free(v);
}

static void test_string_escaping(void) {
    puts("string escaping: quotes, backslash, newline round-trip through parse and write");

    const char *text = "\"line one\\nline \\\"two\\\" with \\\\backslash\\\\\"";
    JsonValue *v = json_parse(text);
    assert(v->type == JSON_STRING);
    assert(strcmp(v->string, "line one\nline \"two\" with \\backslash\\") == 0);

    char *written = json_write(v);
    JsonValue *reparsed = json_parse(written);
    assert(json_deep_equal(v, reparsed));

    free(written);
    json_free(v);
    json_free(reparsed);
}

static void test_unicode_escape(void) {
    puts("\\uXXXX escape decodes to UTF-8");

    /* é = 'e' with acute accent (U+00E9) - the 2-byte UTF-8 branch
       (codepoint < 0x800), UTF-8: 0xC3 0xA9 */
    JsonValue *v = json_parse("\"caf\\u00e9\"");
    assert(v->type == JSON_STRING);
    unsigned char expected[] = { 'c','a','f', 0xC3, 0xA9, '\0' };
    assert(strcmp(v->string, (const char*)expected) == 0);
    json_free(v);

    /* the euro sign (U+20AC) - the 3-byte UTF-8 branch (codepoint >=
       0x800), UTF-8: 0xE2 0x82 0xAC. Previously untested: the only fixed
       case above never reaches a codepoint this large, and the STRESS
       fuzzer can't reach it either, since json_write only ever emits
       \u escapes for control characters (< 0x20), never round-tripping
       a codepoint this size back through \uXXXX form. */
    JsonValue *euro = json_parse("\"\\u20ac5\"");
    assert(euro->type == JSON_STRING);
    unsigned char expected_euro[] = { 0xE2, 0x82, 0xAC, '5', '\0' };
    assert(strcmp(euro->string, (const char*)expected_euro) == 0);

    /* round-trip: json_write emits the already-UTF-8-encoded bytes
       literally (not re-escaped, per the note above), so parsing the
       written text a second time must reproduce the identical string */
    char *written = json_write(euro);
    JsonValue *reparsed = json_parse(written);
    assert(strcmp(reparsed->string, euro->string) == 0);
    free(written);
    json_free(euro);
    json_free(reparsed);
}

static void test_construction_and_roundtrip(void) {
    puts("construction API + round-trip: build a tree, write it, parse it back");

    JsonValue *root = json_object();
    json_object_set(root, "name", json_string("mlp"));
    json_object_set(root, "epochs", json_number(3000));
    JsonValue *sizes = json_array();
    json_array_push(sizes, json_number(2));
    json_array_push(sizes, json_number(4));
    json_array_push(sizes, json_number(1));
    json_object_set(root, "sizes", sizes);

    char *text = json_write(root);
    JsonValue *parsed = json_parse(text);
    assert(json_deep_equal(root, parsed));

    free(text);
    json_free(root);
    json_free(parsed);
}

static void test_duplicate_key_last_wins(void) {
    puts("duplicate object key: last value wins, matching JSON semantics");

    JsonValue *v = json_parse("{\"x\": 1, \"x\": 2}");
    assert(v->n_members == 1);
    assert(json_as_number(json_object_get(v, "x")) == 2);
    json_free(v);
}

static void test_adversarial(void) {
    puts("adversarial: empty object/array, deep nesting, negative/fractional/exponent numbers");

    {
        JsonValue *v = json_parse("{}");
        assert(v->type == JSON_OBJECT && v->n_members == 0);
        json_free(v);
    }
    {
        JsonValue *v = json_parse("[]");
        assert(v->type == JSON_ARRAY && v->n_items == 0);
        json_free(v);
    }
    {
        /* [[[[[1]]]]] - 5 levels of nested single-element arrays */
        JsonValue *v = json_parse("[[[[[1]]]]]");
        JsonValue *cur = v;
        for (int i = 0; i < 5; i++) { assert(cur->type == JSON_ARRAY && cur->n_items == 1); cur = cur->items[0]; }
        assert(cur->type == JSON_NUMBER && cur->number == 1);
        json_free(v);
    }
    {
        JsonValue *v = json_parse("[-3.5, 0, 1e10, -2.5e-3]");
        assert(json_as_number(json_array_get(v, 0)) == -3.5);
        assert(json_as_number(json_array_get(v, 1)) == 0);
        assert(json_as_number(json_array_get(v, 2)) == 1e10);
        assert(json_as_number(json_array_get(v, 3)) == -2.5e-3);
        json_free(v);
    }
}

static void test_file_roundtrip(void) {
    puts("file round-trip: json_write_file then json_parse_file");

    const char *path = "/tmp/clgebra_test.json";
    JsonValue *root = json_object();
    json_object_set(root, "a", json_number(42));
    json_object_set(root, "b", json_string("hello"));
    json_write_file(root, path);

    JsonValue *loaded = json_parse_file(path);
    assert(json_deep_equal(root, loaded));

    json_free(root);
    json_free(loaded);
    remove(path);
}

/* --- STRESS=1: fixed-seed randomized round-trip fuzzing, per this
   project's testing policy (see README's Testing requirements: "use
   randomized/fuzz inputs heavily, but fix the seed... bias them toward
   fragile regions"). Generates random JSON trees biased toward the
   characters/magnitudes most likely to expose an escaping or number-
   formatting bug, writes each, parses it back, and checks structural
   equality against the original - the same round-trip oracle the fixed
   cases above use, just at volume and randomized. --- */

static JsonValue *random_json_value(int max_depth) {
    int max_choice = max_depth > 0 ? 6 : 4; /* leaf-only types once depth is exhausted */
    switch (rand() % max_choice) {
        case 0: return json_null();
        case 1: return json_bool(rand() % 2);
        case 2: {
            /* biased toward fragile magnitudes: zero, negative, fractional,
               very large, very small - not just well-conditioned mid-range values */
            switch (rand() % 5) {
                case 0: return json_number(0.0);
                case 1: return json_number(-((double)(rand() % 100000)) / 7.0);
                case 2: return json_number(((double)(rand() % 100000)) / 3.0);
                case 3: return json_number((rand() % 2 ? 1.0 : -1.0) * 1e300);
                default: return json_number(1e-300);
            }
        }
        case 3: {
            /* biased toward characters that need escaping */
            static const char fragile[] = "\"\\\n\t\rabc \x01\x1f";
            int len = rand() % 20;
            char buf[21];
            for (int i = 0; i < len; i++) buf[i] = fragile[rand() % (sizeof(fragile) - 1)];
            buf[len] = '\0';
            return json_string(buf);
        }
        case 4: {
            JsonValue *arr = json_array();
            int n = rand() % 5;
            for (int i = 0; i < n; i++) json_array_push(arr, random_json_value(max_depth - 1));
            return arr;
        }
        default: {
            JsonValue *obj = json_object();
            int n = rand() % 5;
            for (int i = 0; i < n; i++) {
                char key[16];
                snprintf(key, sizeof key, "k%d", rand() % 1000);
                json_object_set(obj, key, random_json_value(max_depth - 1));
            }
            return obj;
        }
    }
}

static void test_random_roundtrip_stress(void) {
    puts("  random round-trip fuzzing (fixed seed, fragile-biased)");
    srand(42);
    for (int trial = 0; trial < 200; trial++) {
        JsonValue *original = random_json_value(4);
        char *text = json_write(original);
        JsonValue *parsed = json_parse(text);
        assert(json_deep_equal(original, parsed));
        free(text);
        json_free(original);
        json_free(parsed);
    }
    printf("  200 random JSON trees (depth<=4, fragile-biased strings/numbers) round-tripped ok\n");
}

int main(void) {
    test_known_output_parse();
    test_string_escaping();
    test_unicode_escape();
    test_construction_and_roundtrip();
    test_duplicate_key_last_wins();
    test_adversarial();
    test_file_roundtrip();

    if (getenv("STRESS")) test_random_roundtrip_stress();

    puts("test_json: all passed");
    return 0;
}
