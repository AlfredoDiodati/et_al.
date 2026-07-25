#include "../../json.h"

/* Flat-pointer wrappers for ctypes benchmarking (see bench_json.py) - the
   one benchmark pair for json.h. Parse and write are timed separately,
   matching how json.loads/json.dumps are timed independently on the
   Python side: c_json_build_from_text parses once (untimed) and keeps
   the tree in g_tree so c_json_write_only can be timed on its own,
   the same "load once, time the operation alone" split bench_frame.c
   uses for g_df/c_sql_query. */

void c_json_parse_only(const char *text) {
    JsonValue *v = json_parse(text);
    json_free(v);
}

static JsonValue *g_tree = NULL;

void c_json_build_from_text(const char *text) {
    if (g_tree) json_free(g_tree);
    g_tree = json_parse(text);
}

int c_json_write_to_buf(char *out, int cap) {
    char *s = json_write(g_tree);
    int len = (int)strlen(s);
    if (out && len < cap) memcpy(out, s, (size_t)len + 1);
    free(s);
    return len;
}

void c_json_write_only(void) {
    char *s = json_write(g_tree);
    free(s);
}

void c_json_close(void) {
    if (g_tree) json_free(g_tree);
    g_tree = NULL;
}
