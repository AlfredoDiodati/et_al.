#pragma once
#include "frame.h"
#include "../gzip.h"

/* Reader for R's binary serialization format - the format behind both
   .RData (a saved workspace, produced by save()) and .rds (a single
   saved object, produced by saveRDS()), documented in the "R Internals"
   manual's "Serialization Formats" chapter. Core tier (see README's
   "Installation tiers" policy): a data-loading primitive, sitting next
   to frame/csv.h and frame/npy.h, needing frame/frame.h for DataFrame
   plus the project's own gzip.h (save() gzip-compresses by default, and
   this project's dependency policy rules out zlib - see gzip.h's own
   comment).

   Scope, stated so a caller with an unusual .RData file knows where the
   line is: this reads NULL, logical/integer/double/character vectors,
   lists (including data.frames and factors), dim/dimnames/names/class/
   levels attributes, and the two ALTREP classes R actually uses for
   compact integer/double ranges (compact_intseq/compact_realseq - R's
   encoding for any `1:n`/seq_len()/seq_along()-derived vector, since R
   3.5, materialized from their (count, start, step) state into a plain
   vector). It does not read environments, closures, S4 objects, language
   objects/calls, or any other ALTREP class (deferred strings, memory-
   mapped vectors, wrapped vectors, ...). Anything outside this list fails
   a clearly worded assert naming what wasn't understood, rather than
   silently producing wrong data - the same "assert on contract
   violation, not a recoverable error path" convention frame/npy.h's
   header validation already follows.

   df_from_rvalue does not carry a data.frame's row.names into the
   resulting DataFrame's row_names, in any encoding - it is simply not
   wired in, not a format limitation: the common (default, automatically-
   generated) encoding is a plain 2-element INTSXP (c(NA_integer_,
   -nrow)), not ALTREP, so this parser already reads it fine. DataFrame's
   row_names is already optional/NULL-able for exactly this "no labels
   carried over" case. R's reference-deduplication (REFSXP) applies only
   to symbols and environments, never CHARSXP string constants; see
   docs/RDATA_DOCUMENTATION.md's Scope section for how both of these
   claims were checked against real R output.

   RValue mirrors JsonValue's design (json.h) as closely as R's object
   model allows: one flat-ish tagged union covering every supported
   SEXPTYPE, "general-purpose value tree, not tied to a model". Where it
   differs is attributes - dim/dimnames/names/class are promoted to
   named fields on every RValue (matching R itself, where any SEXP can
   carry any of them) rather than nested as a generic key/value nesting;
   any attribute this loader does not recognize is still fully parsed
   (to keep the byte stream aligned) and then discarded. */

typedef enum { RVALUE_NULL, RVALUE_LOGICAL, RVALUE_INTEGER, RVALUE_REAL, RVALUE_STRING, RVALUE_LIST } RValueType;

typedef struct RValue RValue;

struct RValue {
    RValueType type;
    int n; /* vector/list length; 0 for RVALUE_NULL */
    union {
        int *logical; /* RVALUE_LOGICAL - R's 0/1, NA as INT_MIN */
        int *integer; /* RVALUE_INTEGER - NA as INT_MIN */
        mreal *real; /* RVALUE_REAL */
        char **string; /* RVALUE_STRING - n owned strings, NA entries are NULL */
        RValue *list; /* RVALUE_LIST - n owned entries (R's VECSXP: a generic
                          list, or a data.frame when class_name says so) */
    } v;
    char **names; /* optional "names" attribute - n owned strings, NULL if absent */
    int *dim; /* optional "dim" attribute - ndim entries, NULL if absent */
    int ndim;
    char ***dimnames; /* optional "dimnames" attribute - ndim entries; entry i is
                          NULL if that dimension has no names, else dim[i] owned strings */
    char *class_name; /* optional "class" attribute's first element, owned, NULL if absent */
    char **levels; int n_levels; /* optional "levels" attribute (an R factor's category labels,
                                     1-indexed by the RVALUE_INTEGER codes) - n_levels owned
                                     strings, NULL/0 if absent */
};

/* A saved workspace: the name/value pairs a .RData file's top-level
   pairlist holds, in save() order - what R's load() would drop into an
   environment. */
typedef struct {
    char **names; /* n owned strings */
    RValue *values; /* n owned values */
    int n;
} RData;

static inline void rvalue_free(RValue *v) {
    if (v->type == RVALUE_STRING) {
        for (int i = 0; i < v->n; i++) free(v->v.string[i]);
        free(v->v.string);
    } else if (v->type == RVALUE_LIST) {
        for (int i = 0; i < v->n; i++) rvalue_free(&v->v.list[i]);
        free(v->v.list);
    } else if (v->type == RVALUE_LOGICAL || v->type == RVALUE_INTEGER) {
        free(v->v.integer);
    } else if (v->type == RVALUE_REAL) {
        free(v->v.real);
    }
    if (v->names) {
        for (int i = 0; i < v->n; i++) free(v->names[i]);
        free(v->names);
    }
    if (v->dimnames) {
        for (int i = 0; i < v->ndim; i++) {
            if (!v->dimnames[i]) continue;
            for (int j = 0; j < v->dim[i]; j++) free(v->dimnames[i][j]);
            free(v->dimnames[i]);
        }
        free(v->dimnames);
    }
    free(v->dim);
    free(v->class_name);
    if (v->levels) {
        for (int i = 0; i < v->n_levels; i++) free(v->levels[i]);
        free(v->levels);
    }
}

/* --- the SEXPTYPE codes and pseudo-types this parser needs to recognize
   (R Internals, "Serialization Formats"). Only a name for every code
   actually referenced below is given; anything else falls into the
   unsupported-type assert in rdata_read_value. --- */
#define RSEXP_NIL 0
#define RSEXP_SYM 1
#define RSEXP_LIST 2
#define RSEXP_CHAR 9
#define RSEXP_LGL 10
#define RSEXP_INT 13
#define RSEXP_REAL 14
#define RSEXP_STR 16
#define RSEXP_VEC 19
#define RSEXP_EXPR 20
#define RSEXP_ALTREP 238
#define RSEXP_NILVALUE 254
#define RSEXP_REF 255

typedef struct {
    const unsigned char *d;
    size_t len, pos;
    char **ref_names; /* the reference table (R Internals: written only for symbols) -
                          growable array of owned copies, indexed 1-based by REFSXP */
    int ref_n, ref_cap;
} RStream;

static inline unsigned char rdata_read_u8(RStream *s) {
    assert(s->pos < s->len && "rdata: truncated stream");
    return s->d[s->pos++];
}

static inline int32_t rdata_read_i32(RStream *s) {
    assert(s->pos + 4 <= s->len && "rdata: truncated stream");
    uint32_t v = ((uint32_t)s->d[s->pos] << 24) | ((uint32_t)s->d[s->pos + 1] << 16) |
                 ((uint32_t)s->d[s->pos + 2] << 8) | (uint32_t)s->d[s->pos + 3];
    s->pos += 4;
    return (int32_t)v;
}

/* R's XDR stream stores doubles big-endian; this project's target
   platforms (x86/ARM, matching frame/npy.h's own note) are little-endian,
   so the 8 bytes are reversed into a native double before use. */
static inline double rdata_read_f64(RStream *s) {
    assert(s->pos + 8 <= s->len && "rdata: truncated stream");
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = s->d[s->pos + 7 - i];
    s->pos += 8;
    double v;
    memcpy(&v, b, 8);
    return v;
}

static inline void rdata_read_bytes(RStream *s, void *out, size_t n) {
    assert(s->pos + n <= s->len && "rdata: truncated stream");
    memcpy(out, s->d + s->pos, n);
    s->pos += n;
}

static inline void rdata_ref_push(RStream *s, const char *name) {
    if (s->ref_n == s->ref_cap) {
        s->ref_cap = s->ref_cap ? s->ref_cap * 2 : 16;
        s->ref_names = (char**)realloc(s->ref_names, (size_t)s->ref_cap * sizeof(char*));
    }
    s->ref_names[s->ref_n++] = frame_strdup(name);
}

/* Reads a CHARSXP that stands alone as a symbol's printname or a pairlist
   tag - never itself reference-encoded (R Internals: only symbols and
   environments go through the reference table, never their printname
   CHARSXPs), so this always reads a literal length-prefixed string. NA
   (length -1) cannot occur here (a symbol's name is never missing). */
static inline char *rdata_read_charsxp(RStream *s) {
    int32_t flags = rdata_read_i32(s);
    assert((flags & 255) == RSEXP_CHAR && "rdata: expected a CHARSXP");
    int32_t len = rdata_read_i32(s);
    assert(len >= 0 && "rdata: a symbol/tag name cannot be NA");
    char *str = (char*)malloc((size_t)len + 1);
    rdata_read_bytes(s, str, (size_t)len);
    str[len] = '\0';
    return str;
}

/* A CHARSXP element inside a STRSXP can be NA (length -1), unlike
   rdata_read_charsxp above (symbol/tag names, never NA) - this is that
   variant, used only by the RSEXP_STR case in rdata_read_value. */
static inline char *rdata_read_charsxp_element(RStream *s) {
    int32_t flags = rdata_read_i32(s);
    assert((flags & 255) == RSEXP_CHAR && "rdata: expected a CHARSXP inside a character vector");
    int32_t len = rdata_read_i32(s);
    if (len < 0) return NULL; /* NA_STRING */
    char *str = (char*)malloc((size_t)len + 1);
    rdata_read_bytes(s, str, (size_t)len);
    str[len] = '\0';
    return str;
}

/* Reads whatever stands in a pairlist TAG or an attribute name position:
   either a fresh SYMSXP (registered into the reference table for later
   reuse) or a REFSXP pointing back at one already seen. Returns an owned
   copy either way - the ref table keeps its own independent copy, so
   freeing the caller's copy never touches it. */
static inline char *rdata_read_tag_or_symbol(RStream *s) {
    int32_t flags = rdata_read_i32(s);
    int type = flags & 255;
    if (type == RSEXP_REF) {
        int refidx = flags >> 8;
        if (refidx == 0) refidx = rdata_read_i32(s);
        assert(refidx >= 1 && refidx <= s->ref_n && "rdata: invalid symbol back-reference (corrupt stream)");
        return frame_strdup(s->ref_names[refidx - 1]);
    }
    assert(type == RSEXP_SYM && "rdata: expected a symbol (attribute name or saved object's name)");
    char *name = rdata_read_charsxp(s);
    rdata_ref_push(s, name);
    return name;
}

static inline RValue rdata_read_value(RStream *s);

/* Reads the 3-element unnamed pairlist R writes ahead of an ALTREP
   object's serialized state (R Internals: `Rf_cons(class_sym,
   Rf_cons(package_sym, Rf_cons(ScalarInteger(base_type), R_NilValue)))`)
   - unlike an attribute pairlist, these cells carry no TAG, so
   rdata_read_attributes cannot be reused here. The base SEXPTYPE element
   is read (to keep the stream aligned) and discarded: this parser
   dispatches purely on the class name string, not the base type. Returns
   the class and package names (owned) via out-parameters. */
static inline void rdata_read_altrep_class_info(RStream *s, char **class_name_out, char **package_out) {
    int32_t f1 = rdata_read_i32(s);
    assert((f1 & 255) == RSEXP_LIST && !(f1 & (1 << 10)) && "rdata: malformed ALTREP class info (class name cell)");
    *class_name_out = rdata_read_tag_or_symbol(s);

    int32_t f2 = rdata_read_i32(s);
    assert((f2 & 255) == RSEXP_LIST && !(f2 & (1 << 10)) && "rdata: malformed ALTREP class info (package name cell)");
    *package_out = rdata_read_tag_or_symbol(s);

    int32_t f3 = rdata_read_i32(s);
    assert((f3 & 255) == RSEXP_LIST && !(f3 & (1 << 10)) && "rdata: malformed ALTREP class info (base type cell)");
    RValue base_type = rdata_read_value(s);
    rvalue_free(&base_type);

    int32_t f4 = rdata_read_i32(s);
    assert((((f4 & 255) == RSEXP_NIL) || ((f4 & 255) == RSEXP_NILVALUE)) &&
           "rdata: malformed ALTREP class info (expected terminating NIL)");
}

/* Reads the attribute pairlist attached to the value just read (R writes
   it as a LISTSXP chain, tag = attribute name symbol, car = attribute
   value, terminated by NIL/NILVALUE - identical shape to the top-level
   workspace pairlist rdata_read below walks). Promotes the four
   attributes this loader understands (dim, dimnames, names, class) onto
   target by taking ownership of their storage directly (no extra copy);
   any other attribute is parsed in full, to keep the stream aligned for
   whatever follows, and then discarded. */
static inline void rdata_read_attributes(RStream *s, RValue *target) {
    for (;;) {
        int32_t flags = rdata_read_i32(s);
        int type = flags & 255;
        if (type == RSEXP_NIL || type == RSEXP_NILVALUE) return;
        assert(type == RSEXP_LIST && "rdata: malformed attribute list");
        assert(!(flags & (1 << 9)) && "rdata: attributes on an attribute pairlist cell are not supported");
        assert((flags & (1 << 10)) && "rdata: attribute pairlist entry without a name");

        char *attr_name = rdata_read_tag_or_symbol(s);
        RValue attr_val = rdata_read_value(s);

        /* Every steal below clears the source's own n (not just the
           pointer) - rvalue_free's string/list cases iterate 0..n
           freeing each element, so a stolen pointer left under the
           original n dereferences NULL past element 0. Clearing type to
           RVALUE_NULL alongside n keeps the source in a state rvalue_free
           already knows is a fully-freed no-op. */
        if (strcmp(attr_name, "dim") == 0) {
            assert(attr_val.type == RVALUE_INTEGER && "rdata: 'dim' attribute is not an integer vector");
            target->ndim = attr_val.n;
            target->dim = attr_val.v.integer;
            attr_val.v.integer = NULL;
            attr_val.type = RVALUE_NULL;
            attr_val.n = 0;
        } else if (strcmp(attr_name, "names") == 0) {
            assert(attr_val.type == RVALUE_STRING && "rdata: 'names' attribute is not a character vector");
            target->names = attr_val.v.string;
            attr_val.v.string = NULL;
            attr_val.type = RVALUE_NULL;
            attr_val.n = 0;
        } else if (strcmp(attr_name, "dimnames") == 0) {
            assert(attr_val.type == RVALUE_LIST && "rdata: 'dimnames' attribute is not a list");
            target->dimnames = (char***)malloc((size_t)attr_val.n * sizeof(char**));
            for (int i = 0; i < attr_val.n; i++) {
                if (attr_val.v.list[i].type == RVALUE_NULL) {
                    target->dimnames[i] = NULL;
                } else {
                    assert(attr_val.v.list[i].type == RVALUE_STRING && "rdata: a 'dimnames' entry is not a character vector");
                    target->dimnames[i] = attr_val.v.list[i].v.string;
                    attr_val.v.list[i].v.string = NULL;
                    attr_val.v.list[i].type = RVALUE_NULL;
                    attr_val.v.list[i].n = 0;
                }
            }
        } else if (strcmp(attr_name, "class") == 0) {
            assert(attr_val.type == RVALUE_STRING && attr_val.n >= 1 && "rdata: 'class' attribute is not a character vector");
            target->class_name = frame_strdup(attr_val.v.string[0]);
        } else if (strcmp(attr_name, "levels") == 0) {
            assert(attr_val.type == RVALUE_STRING && "rdata: 'levels' attribute is not a character vector");
            target->n_levels = attr_val.n;
            target->levels = attr_val.v.string;
            attr_val.v.string = NULL;
            attr_val.type = RVALUE_NULL;
            attr_val.n = 0;
        }

        free(attr_name);
        rvalue_free(&attr_val);
    }
}

/* Reads one value (whatever sits in a pairlist CAR, a list element, or
   the top-level workspace slot) - the parser's main dispatch, mirroring
   R's own WriteItem: payload first, then (if the HAS_ATTR flag bit is
   set) the attribute pairlist. */
static inline RValue rdata_read_value(RStream *s) {
    int32_t flags = rdata_read_i32(s);
    int type = flags & 255;
    int hasattr = (flags & (1 << 9)) != 0;

    RValue val;
    memset(&val, 0, sizeof val);

    switch (type) {
    case RSEXP_NIL:
    case RSEXP_NILVALUE:
        val.type = RVALUE_NULL;
        return val; /* R_NilValue never carries attributes */

    case RSEXP_LGL:
    case RSEXP_INT: {
        int32_t n = rdata_read_i32(s);
        assert(n >= 0 && "rdata: negative vector length (corrupt stream, or a >2^31-element "
                          "'long vector' - not supported)");
        int *data = (int*)malloc((size_t)n * sizeof(int));
        for (int32_t i = 0; i < n; i++) data[i] = rdata_read_i32(s);
        val.type = (type == RSEXP_LGL) ? RVALUE_LOGICAL : RVALUE_INTEGER;
        val.n = n;
        val.v.integer = data;
        break;
    }
    case RSEXP_REAL: {
        int32_t n = rdata_read_i32(s);
        assert(n >= 0 && "rdata: negative vector length (corrupt stream, or a >2^31-element "
                          "'long vector' - not supported)");
        mreal *data = (mreal*)malloc((size_t)n * sizeof(mreal));
        for (int32_t i = 0; i < n; i++) data[i] = (mreal)rdata_read_f64(s);
        val.type = RVALUE_REAL;
        val.n = n;
        val.v.real = data;
        break;
    }
    case RSEXP_STR: {
        int32_t n = rdata_read_i32(s);
        assert(n >= 0 && "rdata: negative vector length (corrupt stream)");
        char **data = (char**)malloc((size_t)n * sizeof(char*));
        for (int32_t i = 0; i < n; i++) data[i] = rdata_read_charsxp_element(s);
        val.type = RVALUE_STRING;
        val.n = n;
        val.v.string = data;
        break;
    }
    case RSEXP_VEC:
    case RSEXP_EXPR: {
        int32_t n = rdata_read_i32(s);
        assert(n >= 0 && "rdata: negative vector length (corrupt stream)");
        RValue *data = (RValue*)malloc((size_t)n * sizeof(RValue));
        for (int32_t i = 0; i < n; i++) data[i] = rdata_read_value(s);
        val.type = RVALUE_LIST;
        val.n = n;
        val.v.list = data;
        break;
    }
    case RSEXP_ALTREP: {
        /* An ALTREP object serializes as class info, then a class-specific
           "state", then - unconditionally, regardless of this node's own
           HAS_ATTR bit - the object's own attribute pairlist. The bit
           being unset here even when a trailing attribute pairlist
           follows is not a bug: ALTREP's third component is always
           present, independent of that bit. Only the two ALTREP classes
           R uses for compact integer/double ranges (`1:n`, `seq_len`,
           `seq_along`, and similar) are materialized, by expanding their
           3-double (count, start, step) state into a plain vector; every
           other ALTREP class (deferred strings, memory-mapped vectors,
           wrapped vectors, ...) is a clearly named unsupported case
           rather than a silent misread. */
        char *class_name, *package;
        rdata_read_altrep_class_info(s, &class_name, &package);
        RValue state = rdata_read_value(s);

        if (strcmp(class_name, "compact_intseq") == 0 || strcmp(class_name, "compact_realseq") == 0) {
            assert(state.type == RVALUE_REAL && state.n == 3 &&
                   "rdata: malformed compact-sequence ALTREP state (expected a 3-element double vector)");
            double count = state.v.real[0], start = state.v.real[1], step = state.v.real[2];
            int n = (int)count;
            assert((double)n == count && n >= 0 && "rdata: compact sequence length out of range");

            if (strcmp(class_name, "compact_intseq") == 0) {
                int *data = (int*)malloc((size_t)n * sizeof(int));
                for (int i = 0; i < n; i++) data[i] = (int)(start + (double)i * step);
                val.type = RVALUE_INTEGER;
                val.n = n;
                val.v.integer = data;
            } else {
                mreal *data = (mreal*)malloc((size_t)n * sizeof(mreal));
                for (int i = 0; i < n; i++) data[i] = (mreal)(start + (double)i * step);
                val.type = RVALUE_REAL;
                val.n = n;
                val.v.real = data;
            }
        } else {
            fprintf(stderr, "rdata: unsupported ALTREP class '%s' (package '%s') at byte offset %zu\n",
                    class_name, package, s->pos);
            free(class_name);
            free(package);
            rvalue_free(&state);
            assert(0 && "rdata: unsupported ALTREP class - only 'compact_intseq'/'compact_realseq' "
                         "(R's encoding for '1:n'-style integer/double ranges) are materialized; every "
                         "other ALTREP class is not");
        }

        rvalue_free(&state);
        free(class_name);
        free(package);
        rdata_read_attributes(s, &val); /* unconditional - see comment above */
        return val;
    }
    default:
        fprintf(stderr, "rdata: unsupported R SEXP type code %d at byte offset %zu (see frame/rdata.h's "
                         "header comment for what is and isn't supported)\n", type, s->pos - 4);
        assert(0 && "rdata: unsupported R object type - only NULL, logical/integer/double/character "
                     "vectors, and lists (including data.frames) are supported; environments, "
                     "closures, S4 objects, language objects, and ALTREP-encoded vectors are not");
        return val;
    }

    if (hasattr) rdata_read_attributes(s, &val);
    return val;
}

/* Reads the "X\n" + version header shared by every R serialization
   stream (R Internals, "Serialization Formats"): a format version, the
   writer's and minimum reader's R version (packed, not needed here), and
   - format version 3 only - a length-prefixed native encoding name
   (also not needed: strings are copied as raw bytes either way). Leaves
   s->pos at the start of the actual object stream. */
static inline void rdata_read_header(RStream *s) {
    assert(s->len >= 7 && s->d[5] == 'X' && s->d[6] == '\n' &&
           "rdata: unsupported stream type (only XDR/binary streams are supported, not ASCII or raw)");
    s->pos = 7;
    int32_t format_version = rdata_read_i32(s);
    rdata_read_i32(s); /* writer R version */
    rdata_read_i32(s); /* minimum reader R version */
    assert((format_version == 2 || format_version == 3) && "rdata: unsupported serialization format version");
    if (format_version == 3) {
        int32_t enc_len = rdata_read_i32(s);
        assert(s->pos + (size_t)enc_len <= s->len && "rdata: truncated header");
        s->pos += (size_t)enc_len;
    }
}

/* Reads a whole .RData workspace file: gzip-decompresses if needed (plain
   save() output always is; save(..., compress = FALSE) is not, so both
   are accepted - see gzip_is_gzip), then walks the top-level pairlist
   (magic "RDX2\n"/"RDX3\n", tag = each saved object's name, car = its
   value) into an RData. */
static inline RData rdata_read(const char *path) {
    long file_size;
    unsigned char *raw = (unsigned char*)frame_read_file(path, &file_size);

    unsigned char *buf;
    size_t buf_len;
    if (gzip_is_gzip(raw, (size_t)file_size)) {
        buf = gzip_inflate(raw, (size_t)file_size, &buf_len);
        free(raw);
    } else {
        buf = raw;
        buf_len = (size_t)file_size;
    }

    assert(buf_len >= 5 && (memcmp(buf, "RDX2\n", 5) == 0 || memcmp(buf, "RDX3\n", 5) == 0) &&
           "rdata: not an R workspace file (bad magic - expected 'RDX2\\n' or 'RDX3\\n')");

    RStream s = { buf, buf_len, 0, NULL, 0, 0 };
    rdata_read_header(&s);

    RData result = { NULL, NULL, 0 };
    int cap = 0;
    for (;;) {
        int32_t flags = rdata_read_i32(&s);
        int type = flags & 255;
        if (type == RSEXP_NIL || type == RSEXP_NILVALUE) break;
        assert(type == RSEXP_LIST && "rdata: malformed workspace (expected the top-level pairlist)");
        assert(!(flags & (1 << 9)) && "rdata: attributes on a saved workspace's pairlist cell are not supported");
        assert((flags & (1 << 10)) && "rdata: a saved object has no name");

        char *name = rdata_read_tag_or_symbol(&s);
        RValue value = rdata_read_value(&s);

        if (result.n == cap) {
            cap = cap ? cap * 2 : 4;
            result.names = (char**)realloc(result.names, (size_t)cap * sizeof(char*));
            result.values = (RValue*)realloc(result.values, (size_t)cap * sizeof(RValue));
        }
        result.names[result.n] = name;
        result.values[result.n] = value;
        result.n++;
    }

    for (int i = 0; i < s.ref_n; i++) free(s.ref_names[i]);
    free(s.ref_names);
    free(buf);
    return result;
}

/* Looks up a saved object by name (the name it was assigned in R before
   save()). Asserts if absent - matching frame_col_lookup's "a lookup by
   name is a contract, not a maybe" convention elsewhere in frame/. */
static inline RValue *rdata_get(const RData *d, const char *name) {
    for (int i = 0; i < d->n; i++) if (strcmp(d->names[i], name) == 0) return &d->values[i];
    assert(0 && "rdata: no saved object with this name");
    return NULL;
}

static inline void rdata_free(RData *d) {
    for (int i = 0; i < d->n; i++) { free(d->names[i]); rvalue_free(&d->values[i]); }
    free(d->names);
    free(d->values);
}

/* --- conversion into this project's own numeric types - the actual
   point of reading an .RData file being to hand the data to
   linalg/decomp.h, linalg/solver.h, ad.h, etc. --- */

/* Converts a numeric vector or 2D matrix into a Mat, transposing R's
   column-major storage into Mat's row-major storage (linalg/mat.h's AT
   macro) element by element - not a memcpy, unlike frame/npy.h's loader,
   because the two formats disagree on storage order rather than merely
   on precision. A vector with no 'dim' attribute (ndim == 0) is treated
   as an n x 1 column, matching R's own "a vector is a length-n object,
   not implicitly a row or column" semantics resolved the same way
   df_add_numeric_col's Vec parameter already resolves it. */
static inline Mat rvalue_to_mat(const RValue *v) {
    assert((v->type == RVALUE_REAL || v->type == RVALUE_INTEGER || v->type == RVALUE_LOGICAL) &&
           "rdata: rvalue_to_mat requires a numeric (double/integer/logical) vector or matrix");
    assert(v->ndim <= 2 &&
           "rdata: rvalue_to_mat only handles vectors and 2D matrices - for a higher-dimensional "
           "array, take one 2D slice at a time (see rvalue_array_slice_df for the 3D case)");

    int r = (v->ndim == 2) ? v->dim[0] : v->n;
    int c = (v->ndim == 2) ? v->dim[1] : 1;

    Mat m = mat_new(r, c);
    for (int j = 0; j < c; j++) {
        for (int i = 0; i < r; i++) {
            int idx = i + j * r; /* R arrays are stored column-major */
            AT(m, i, j) = (v->type == RVALUE_REAL) ? v->v.real[idx] : (mreal)v->v.integer[idx];
        }
    }
    return m;
}

/* Converts an R data.frame (a named RVALUE_LIST with class "data.frame")
   into a DataFrame - the natural counterpart to frame/csv.h's loader for
   data that started life in R. Row labels (R's row.names attribute) are
   not carried over - not because they are hard to read (verified against
   real R 4.3.3 output during development: a data.frame's default,
   automatically-generated row.names is a plain 2-element INTSXP,
   `c(NA_integer_, -nrow)`, not ALTREP, and this parser already reads it
   without trouble), but because it is simply not wired into DataFrame's
   `row_names` - both the default form above and an explicit character
   row.names vector are silently dropped either way. DataFrame's
   `row_names` is already optional for exactly this "no labels carried
   over" case.

   A factor column (class "factor": an RVALUE_INTEGER of 1-based codes
   plus a "levels" character vector, both read generically by
   rdata_read_attributes - see the `levels`/`n_levels` fields on RValue)
   is converted to its category *labels*, not its raw integer codes -
   the labels are what a factor actually represents; the codes are an
   implementation detail of how R stores it. A code outside [1, n_levels]
   (R's NA_INTEGER, or any other out-of-range value) becomes the string
   "NA", matching the convention frame/csv.h's own NA handling already
   uses elsewhere in this project. */
static inline DataFrame df_from_rvalue(const RValue *v) {
    assert(v->type == RVALUE_LIST && v->class_name && strcmp(v->class_name, "data.frame") == 0 &&
           "rdata: df_from_rvalue requires an R data.frame (a named list with class 'data.frame')");
    assert(v->names && "rdata: data.frame has no column names");
    assert(v->n > 0 && "rdata: data.frame has no columns");

    int nrow = v->v.list[0].n;
    DataFrame df = df_new(nrow);
    for (int j = 0; j < v->n; j++) {
        const RValue *col = &v->v.list[j];
        assert(col->n == nrow && "rdata: data.frame columns have inconsistent lengths");

        int is_factor = col->type == RVALUE_INTEGER && col->class_name && strcmp(col->class_name, "factor") == 0;
        if (is_factor) {
            assert(col->levels && "rdata: a factor column has no 'levels' attribute");
            char **filled = (char**)malloc((size_t)nrow * sizeof(char*));
            for (int i = 0; i < nrow; i++) {
                int code = col->v.integer[i];
                filled[i] = (code >= 1 && code <= col->n_levels) ? col->levels[code - 1] : "NA";
            }
            df_add_string_col(&df, v->names[j], (const char *const *)filled);
            free(filled);
        } else if (col->type == RVALUE_REAL || col->type == RVALUE_INTEGER || col->type == RVALUE_LOGICAL) {
            Vec vcol = mat_new(nrow, 1);
            for (int i = 0; i < nrow; i++)
                vcol.d[i] = (col->type == RVALUE_REAL) ? col->v.real[i] : (mreal)col->v.integer[i];
            df_add_numeric_col(&df, v->names[j], vcol);
            mat_free(vcol);
        } else if (col->type == RVALUE_STRING) {
            char **filled = (char**)malloc((size_t)nrow * sizeof(char*));
            for (int i = 0; i < nrow; i++) filled[i] = col->v.string[i] ? col->v.string[i] : "NA";
            df_add_string_col(&df, v->names[j], (const char *const *)filled);
            free(filled);
        } else {
            assert(0 && "rdata: unsupported data.frame column type (only numeric, character, and factor columns are supported)");
        }
    }
    return df;
}

/* Slices a 3D double array (dim = c(d0, d1, d2), the shape save()
   produces for something like an 8-variable, 600-observation, 108-run
   panel) at a fixed index k along the third dimension, returning it as a
   d1-row x d0-column DataFrame - column names taken from dimnames[[1]]
   when present (df_from_matrix's own NULL fallback to "col0".. otherwise
   applies unchanged). The natural way to pull one run/series/replicate
   out of a saved Monte Carlo panel and hand it to the rest of the
   library one 2D slice at a time. */
static inline DataFrame rvalue_array_slice_df(const RValue *v, int k) {
    assert(v->type == RVALUE_REAL && "rdata: rvalue_array_slice_df requires a double array");
    assert(v->ndim == 3 && "rdata: rvalue_array_slice_df only handles a 3D array (variables x observations x slices)");
    int d0 = v->dim[0], d1 = v->dim[1], d2 = v->dim[2];
    assert(k >= 0 && k < d2 && "rdata: slice index out of range");

    Mat m = mat_new(d1, d0);
    for (int j = 0; j < d0; j++)
        for (int i = 0; i < d1; i++)
            AT(m, i, j) = v->v.real[j + i * d0 + k * d0 * d1];

    char **col_names = (v->dimnames && v->dimnames[0]) ? v->dimnames[0] : NULL;
    DataFrame df = df_from_matrix(m, (const char *const *)col_names);
    mat_free(m);
    return df;
}

/* --- one-shot convenience API: everything above this point is the
   primitive layer (RData/RValue plus the three conversions), for a
   caller that genuinely needs to inspect a workspace's shape or handle
   more than one saved object. Most callers do not - a prototype or
   application script almost always wants exactly one specific object
   from one specific file, converted straight into something the rest of
   this library already works with, and shouldn't need to manage
   RData's lifetime (rdata_read/rdata_free) or think about RValue at all
   to get it. df_read_rdata/df_read_rdata_slice below are that path: the
   same "df_read_<format>(path) -> DataFrame, done" shape frame/csv.h and
   frame/npy.h already establish, extended with the one extra parameter
   an .RData file's multi-object nature actually requires. --- */

/* Reads path, extracts one saved object, and converts it directly into a
   DataFrame - handling both shapes df_read_rdata's caller is likely to
   have on hand automatically: an R data.frame (df_from_rvalue), or a
   plain vector/2D matrix (rvalue_to_mat, wrapped via df_from_matrix -
   column names come from the matrix's own dimnames[[2]], R's column-name
   slot, when it has them). object_name selects which saved object;
   pass NULL when the file holds exactly one (asserted) to skip naming it
   at the call site entirely. Asserts with a message pointing at the
   right lower-level call for anything this can't handle automatically:
   a list that isn't a data.frame, or an array with more than 2
   dimensions (see df_read_rdata_slice for that specific, common case). */
static inline DataFrame df_read_rdata(const char *path, const char *object_name) {
    RData d = rdata_read(path);
    RValue *v;
    if (object_name) {
        v = rdata_get(&d, object_name);
    } else {
        assert(d.n == 1 &&
               "rdata: df_read_rdata needs an explicit object_name - this file holds more than one saved object");
        v = &d.values[0];
    }

    DataFrame df;
    if (v->type == RVALUE_LIST && v->class_name && strcmp(v->class_name, "data.frame") == 0) {
        df = df_from_rvalue(v);
    } else if ((v->type == RVALUE_REAL || v->type == RVALUE_INTEGER || v->type == RVALUE_LOGICAL) && v->ndim <= 2) {
        Mat m = rvalue_to_mat(v);
        char **col_names = (v->ndim == 2 && v->dimnames && v->dimnames[1]) ? v->dimnames[1] : NULL;
        df = df_from_matrix(m, (const char *const *)col_names);
        mat_free(m);
    } else {
        assert(0 && "rdata: df_read_rdata only handles a data.frame or a vector/2D matrix - for a "
                     "3D array see df_read_rdata_slice, and for anything else use rdata_read/rdata_get "
                     "directly on the RValue");
        df = df_new(0); /* unreachable past the assert above; keeps every path returning a value */
    }

    rdata_free(&d);
    return df;
}

/* Reads path, extracts one saved 3D double array, and returns its k-th
   slice (along the array's third dimension) as a DataFrame in one call -
   object_name works exactly as in df_read_rdata (NULL for "the file's
   only object"). The one-call version of
   rdata_read + rdata_get + rvalue_array_slice_df for the common case of
   pulling a single run/series/replicate out of a saved Monte Carlo
   panel without holding onto the RData at all. */
static inline DataFrame df_read_rdata_slice(const char *path, const char *object_name, int k) {
    RData d = rdata_read(path);
    RValue *v;
    if (object_name) {
        v = rdata_get(&d, object_name);
    } else {
        assert(d.n == 1 &&
               "rdata: df_read_rdata_slice needs an explicit object_name - this file holds more than one saved object");
        v = &d.values[0];
    }

    DataFrame df = rvalue_array_slice_df(v, k);
    rdata_free(&d);
    return df;
}
