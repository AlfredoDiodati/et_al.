#pragma once
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* Minimal JSON value tree: parse, build, write. Core tier (see README's
   "Installation tiers" policy) - a general-purpose serialization utility,
   not tied to any model. Written for this project's stated use case -
   saving/loading parameters (hyperparameters, fit diagnostics, config) -
   not as a DataFrame data-loading format the way frame/csv.h etc. are;
   that's why this lives at the root as a standalone file (matching ad.h's
   "core, general-purpose tool" placement) rather than under frame/.

   Deliberately has zero dependency on mat.h/mreal: a parameter is
   typically a standalone scalar (a learning rate, an epoch count, a layer
   width), not part of a Mat computation, so JSON_NUMBER stores a plain
   double - this file's precision is not tied to -DMAT_DOUBLE at all.

   Not a union internally (every JsonValue carries every field, most
   unused depending on .type) - JSON documents in this project's actual
   use case (parameter files) are small, so the memory this wastes is
   irrelevant, and a flat struct is far simpler than get right than a
   C union of heterogeneous owned-pointer types. */

typedef enum { JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT } JsonType;

typedef struct JsonValue {
    JsonType type;
    int boolean;                  /* JSON_BOOL */
    double number;                  /* JSON_NUMBER */
    char *string;                    /* JSON_STRING - owned */
    struct JsonValue **items; int n_items;   /* JSON_ARRAY - owned array of owned pointers */
    char **keys; struct JsonValue **values; int n_members; /* JSON_OBJECT - parallel owned arrays */
} JsonValue;

static inline char *json_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char*)malloc(n);
    memcpy(p, s, n);
    return p;
}

static inline JsonValue *json_alloc(JsonType type) {
    JsonValue *v = (JsonValue*)calloc(1, sizeof(JsonValue));
    v->type = type;
    return v;
}

/* --- construction: build a tree to write out. json_array_push/
   json_object_set take ownership of the value pointer passed in (no
   copy) - the standard "build it, then hand it to the container"
   pattern; json_object_set copies the key string instead, since keys are
   typically string literals the caller keeps using. --- */

static inline JsonValue *json_null(void) { return json_alloc(JSON_NULL); }
static inline JsonValue *json_bool(int b) { JsonValue *v = json_alloc(JSON_BOOL); v->boolean = b; return v; }
static inline JsonValue *json_number(double n) { JsonValue *v = json_alloc(JSON_NUMBER); v->number = n; return v; }
static inline JsonValue *json_string(const char *s) { JsonValue *v = json_alloc(JSON_STRING); v->string = json_strdup(s); return v; }
static inline JsonValue *json_array(void) { return json_alloc(JSON_ARRAY); }

static inline void json_free(JsonValue *v) {
    if (!v) return;
    if (v->type == JSON_STRING) {
        free(v->string);
    } else if (v->type == JSON_ARRAY) {
        for (int i = 0; i < v->n_items; i++) json_free(v->items[i]);
        free(v->items);
    } else if (v->type == JSON_OBJECT) {
        for (int i = 0; i < v->n_members; i++) { free(v->keys[i]); json_free(v->values[i]); }
        free(v->keys);
        free(v->values);
    }
    free(v);
}

static inline void json_array_push(JsonValue *arr, JsonValue *item) {
    assert(arr->type == JSON_ARRAY);
    arr->items = (JsonValue**)realloc(arr->items, (size_t)(arr->n_items + 1) * sizeof(JsonValue*));
    arr->items[arr->n_items++] = item;
}

static inline JsonValue *json_object(void) { return json_alloc(JSON_OBJECT); }

/* Sets key to value, replacing (and freeing) any existing value for the
   same key - last value for a key wins, matching standard JSON object
   semantics and this file's own parser (a duplicate key while parsing
   goes through this same function). */
static inline void json_object_set(JsonValue *obj, const char *key, JsonValue *value) {
    assert(obj->type == JSON_OBJECT);
    for (int i = 0; i < obj->n_members; i++) {
        if (strcmp(obj->keys[i], key) == 0) {
            json_free(obj->values[i]);
            obj->values[i] = value;
            return;
        }
    }
    obj->keys = (char**)realloc(obj->keys, (size_t)(obj->n_members + 1) * sizeof(char*));
    obj->values = (JsonValue**)realloc(obj->values, (size_t)(obj->n_members + 1) * sizeof(JsonValue*));
    obj->keys[obj->n_members] = json_strdup(key);
    obj->values[obj->n_members] = value;
    obj->n_members++;
}

/* --- accessors: all assert on type mismatch (contract violation, not a
   recoverable error path - same convention as linalg/decomp.h/solver.h). --- */

static inline JsonValue *json_object_get(const JsonValue *obj, const char *key) {
    assert(obj->type == JSON_OBJECT);
    for (int i = 0; i < obj->n_members; i++)
        if (strcmp(obj->keys[i], key) == 0) return obj->values[i];
    return NULL;
}
static inline double json_as_number(const JsonValue *v) { assert(v->type == JSON_NUMBER); return v->number; }
static inline const char *json_as_string(const JsonValue *v) { assert(v->type == JSON_STRING); return v->string; }
static inline int json_as_bool(const JsonValue *v) { assert(v->type == JSON_BOOL); return v->boolean; }
static inline int json_array_len(const JsonValue *v) { assert(v->type == JSON_ARRAY); return v->n_items; }
static inline JsonValue *json_array_get(const JsonValue *v, int i) {
    assert(v->type == JSON_ARRAY); assert(i >= 0 && i < v->n_items);
    return v->items[i];
}

/* --- parsing --- */

typedef struct { const char *p; } JsonParser;

static inline void json_skip_ws(JsonParser *jp) {
    while (*jp->p == ' ' || *jp->p == '\t' || *jp->p == '\n' || *jp->p == '\r') jp->p++;
}

/* Parses a "..." string starting at jp->p (which must point at the
   opening quote), handling \", \\, \/, \n, \t, \r, \b, \f, and \uXXXX
   (encoded as UTF-8; only BMP code points - no surrogate-pair handling
   for characters outside it, sufficient for parameter-file text, not a
   full Unicode-in-JSON implementation). Returns a malloc'd C string. */
static inline char *json_parse_string_raw(JsonParser *jp) {
    assert(*jp->p == '"' && "json: expected string");
    jp->p++;
    size_t cap = 32, len = 0;
    char *buf = (char*)malloc(cap);
#define JSTR_PUSH(c) do { if (len + 1 >= cap) { cap *= 2; buf = (char*)realloc(buf, cap); } buf[len++] = (char)(c); } while (0)
    while (*jp->p != '"') {
        assert(*jp->p != '\0' && "json: unterminated string");
        if (*jp->p == '\\') {
            jp->p++;
            char esc = *jp->p;
            if (esc == 'u') {
                jp->p++;
                unsigned cp = 0;
                for (int k = 0; k < 4; k++) {
                    char h = *jp->p++;
                    unsigned digit;
                    if (h >= '0' && h <= '9') digit = (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') digit = (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') digit = (unsigned)(h - 'A' + 10);
                    else { assert(0 && "json: bad \\u escape"); digit = 0; }
                    cp = (cp << 4) | digit;
                }
                if (cp < 0x80) {
                    JSTR_PUSH(cp);
                } else if (cp < 0x800) {
                    JSTR_PUSH(0xC0 | (cp >> 6));
                    JSTR_PUSH(0x80 | (cp & 0x3F));
                } else {
                    JSTR_PUSH(0xE0 | (cp >> 12));
                    JSTR_PUSH(0x80 | ((cp >> 6) & 0x3F));
                    JSTR_PUSH(0x80 | (cp & 0x3F));
                }
            } else {
                char out;
                switch (esc) {
                    case '"': out = '"'; break;
                    case '\\': out = '\\'; break;
                    case '/': out = '/'; break;
                    case 'n': out = '\n'; break;
                    case 't': out = '\t'; break;
                    case 'r': out = '\r'; break;
                    case 'b': out = '\b'; break;
                    case 'f': out = '\f'; break;
                    default: assert(0 && "json: bad escape"); out = 0;
                }
                JSTR_PUSH(out);
                jp->p++;
            }
        } else {
            JSTR_PUSH(*jp->p);
            jp->p++;
        }
    }
    jp->p++; /* closing quote */
    buf[len] = '\0'; /* always room: JSTR_PUSH keeps cap > len */
#undef JSTR_PUSH
    return buf;
}

/* strtod is more lenient than strict JSON number grammar (accepts a
   leading '+', leading zeros, hex floats, ...) - a deliberate
   simplification, not a hidden bug: this file's own writer never
   produces anything strtod would accept but JSON wouldn't, so round-trip
   correctness is unaffected; only reading slightly-malformed external
   JSON differs from a strict-grammar parser. See docs/JSON_DOCUMENTATION.md. */
static inline double json_parse_number_raw(JsonParser *jp) {
    char *end;
    double v = strtod(jp->p, &end);
    assert(end != jp->p && "json: malformed number");
    jp->p = end;
    return v;
}

static inline JsonValue *json_parse_value(JsonParser *jp) {
    json_skip_ws(jp);
    char c = *jp->p;
    if (c == '"') {
        JsonValue *v = json_alloc(JSON_STRING);
        v->string = json_parse_string_raw(jp);
        return v;
    }
    if (c == '{') {
        jp->p++;
        JsonValue *v = json_object();
        json_skip_ws(jp);
        if (*jp->p == '}') { jp->p++; return v; }
        for (;;) {
            json_skip_ws(jp);
            assert(*jp->p == '"' && "json: expected string key in object");
            char *key = json_parse_string_raw(jp);
            json_skip_ws(jp);
            assert(*jp->p == ':' && "json: expected ':' after object key");
            jp->p++;
            JsonValue *val = json_parse_value(jp);
            json_object_set(v, key, val);
            free(key); /* json_object_set copies the key */
            json_skip_ws(jp);
            if (*jp->p == ',') { jp->p++; continue; }
            assert(*jp->p == '}' && "json: expected ',' or '}' in object");
            jp->p++;
            break;
        }
        return v;
    }
    if (c == '[') {
        jp->p++;
        JsonValue *v = json_array();
        json_skip_ws(jp);
        if (*jp->p == ']') { jp->p++; return v; }
        for (;;) {
            json_array_push(v, json_parse_value(jp));
            json_skip_ws(jp);
            if (*jp->p == ',') { jp->p++; continue; }
            assert(*jp->p == ']' && "json: expected ',' or ']' in array");
            jp->p++;
            break;
        }
        return v;
    }
    if (c == 't') {
        assert(strncmp(jp->p, "true", 4) == 0 && "json: malformed literal");
        jp->p += 4;
        JsonValue *v = json_alloc(JSON_BOOL); v->boolean = 1; return v;
    }
    if (c == 'f') {
        assert(strncmp(jp->p, "false", 5) == 0 && "json: malformed literal");
        jp->p += 5;
        JsonValue *v = json_alloc(JSON_BOOL); v->boolean = 0; return v;
    }
    if (c == 'n') {
        assert(strncmp(jp->p, "null", 4) == 0 && "json: malformed literal");
        jp->p += 4;
        return json_alloc(JSON_NULL);
    }
    assert((c == '-' || (c >= '0' && c <= '9')) && "json: unexpected character");
    JsonValue *v = json_alloc(JSON_NUMBER);
    v->number = json_parse_number_raw(jp);
    return v;
}

/* Parses text as a single JSON value (object, array, string, number,
   bool, or null). Asserts on any malformed input or trailing characters
   after the top-level value - a contract violation, not a recoverable
   error path (see linalg/decomp.h's "Contract" section). */
static inline JsonValue *json_parse(const char *text) {
    JsonParser jp; jp.p = text;
    JsonValue *v = json_parse_value(&jp);
    json_skip_ws(&jp);
    assert(*jp.p == '\0' && "json: trailing characters after top-level value");
    return v;
}

static inline JsonValue *json_parse_file(const char *path) {
    FILE *f = fopen(path, "rb");
    assert(f && "json: could not open file");
    assert(fseek(f, 0, SEEK_END) == 0);
    long size = ftell(f);
    assert(size >= 0);
    assert(fseek(f, 0, SEEK_SET) == 0);
    char *buf = (char*)malloc((size_t)size + 1);
    size_t got = fread(buf, 1, (size_t)size, f);
    assert(got == (size_t)size);
    buf[size] = '\0';
    fclose(f);
    JsonValue *v = json_parse(buf);
    free(buf);
    return v;
}

/* --- writing --- */

typedef struct { char *buf; size_t len, cap; } JsonWriteBuf;

static inline void jwb_push_str(JsonWriteBuf *b, const char *s, size_t n) {
    if (b->len + n + 1 > b->cap) {
        while (b->len + n + 1 > b->cap) b->cap *= 2;
        b->buf = (char*)realloc(b->buf, b->cap);
    }
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}
static inline void jwb_push_char(JsonWriteBuf *b, char c) { jwb_push_str(b, &c, 1); }

static inline void json_write_string_escaped(JsonWriteBuf *b, const char *s) {
    jwb_push_char(b, '"');
    for (const unsigned char *p = (const unsigned char*)s; *p; p++) {
        switch (*p) {
            case '"':  jwb_push_str(b, "\\\"", 2); break;
            case '\\': jwb_push_str(b, "\\\\", 2); break;
            case '\n': jwb_push_str(b, "\\n", 2); break;
            case '\t': jwb_push_str(b, "\\t", 2); break;
            case '\r': jwb_push_str(b, "\\r", 2); break;
            case '\b': jwb_push_str(b, "\\b", 2); break;
            case '\f': jwb_push_str(b, "\\f", 2); break;
            default:
                if (*p < 0x20) {
                    char esc[8];
                    int n = snprintf(esc, sizeof esc, "\\u%04x", *p);
                    jwb_push_str(b, esc, (size_t)n);
                } else {
                    jwb_push_char(b, (char)*p);
                }
        }
    }
    jwb_push_char(b, '"');
}

static inline void json_write_value(JsonWriteBuf *b, const JsonValue *v) {
    switch (v->type) {
        case JSON_NULL:
            jwb_push_str(b, "null", 4);
            break;
        case JSON_BOOL:
            v->boolean ? jwb_push_str(b, "true", 4) : jwb_push_str(b, "false", 5);
            break;
        case JSON_NUMBER: {
            /* %.17g: double's shortest-round-trip digit count (see
               json_deep_equal's use of exact == in tests/correctness/
               test_json.c - this is what makes that correct). */
            char num[32];
            int n = snprintf(num, sizeof num, "%.17g", v->number);
            jwb_push_str(b, num, (size_t)n);
            break;
        }
        case JSON_STRING:
            json_write_string_escaped(b, v->string);
            break;
        case JSON_ARRAY:
            jwb_push_char(b, '[');
            for (int i = 0; i < v->n_items; i++) {
                if (i) jwb_push_char(b, ',');
                json_write_value(b, v->items[i]);
            }
            jwb_push_char(b, ']');
            break;
        case JSON_OBJECT:
            jwb_push_char(b, '{');
            for (int i = 0; i < v->n_members; i++) {
                if (i) jwb_push_char(b, ',');
                json_write_string_escaped(b, v->keys[i]);
                jwb_push_char(b, ':');
                json_write_value(b, v->values[i]);
            }
            jwb_push_char(b, '}');
            break;
    }
}

/* Serializes v to a malloc'd, null-terminated compact JSON string (no
   pretty-printing/indentation - this is for machine round-tripping, not
   human editing). Caller must free() it. */
static inline char *json_write(const JsonValue *v) {
    JsonWriteBuf b;
    b.cap = 64; b.len = 0; b.buf = (char*)malloc(b.cap); b.buf[0] = '\0';
    json_write_value(&b, v);
    return b.buf;
}

static inline void json_write_file(const JsonValue *v, const char *path) {
    char *s = json_write(v);
    FILE *f = fopen(path, "wb");
    assert(f && "json: could not open file for writing");
    fputs(s, f);
    fclose(f);
    free(s);
}
