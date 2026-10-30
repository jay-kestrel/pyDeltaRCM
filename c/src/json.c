#include "json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p;
    const char *end;
} P;

static jval *jnew(jtype t)
{
    jval *v = calloc(1, sizeof *v);
    if (v) v->t = t;
    return v;
}

void json_free(jval *v)
{
    if (!v) return;
    free(v->s);
    for (int i = 0; i < v->n; i++) {
        json_free(v->items[i]);
        if (v->keys) free(v->keys[i]);
    }
    free(v->items);
    free(v->keys);
    free(v);
}

static void skip_ws(P *p)
{
    while (p->p < p->end && isspace((unsigned char)*p->p)) p->p++;
}

static char *parse_string(P *p)
{
    if (p->p >= p->end || *p->p != '"') return NULL;
    p->p++;
    const char *start = p->p;
    while (p->p < p->end && *p->p != '"') {
        if (*p->p == '\\' && p->p + 1 < p->end) p->p++;
        p->p++;
    }
    if (p->p >= p->end) return NULL;
    size_t len = (size_t)(p->p - start);
    p->p++; /* closing quote */
    char *s = malloc(len + 1);
    /* unescape the two escapes that can appear in our configs */
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        char c = start[i];
        if (c == '\\' && i + 1 < len) {
            char e = start[i + 1];
            if (e == 'n') s[j++] = '\n';
            else if (e == 't') s[j++] = '\t';
            else s[j++] = e;
            i++;
        } else {
            s[j++] = c;
        }
    }
    s[j] = '\0';
    return s;
}

static jval *parse_value(P *p);

static jval *parse_object(P *p)
{
    jval *o = jnew(J_OBJ);
    if (!o) return NULL;
    p->p++; /* { */
    size_t cap = 0;
    for (;;) {
        skip_ws(p);
        if (p->p >= p->end) { json_free(o); return NULL; }
        if (*p->p == '}') { p->p++; return o; }
        if (*p->p == ',') { p->p++; continue; }
        char *key = parse_string(p);
        if (!key) { json_free(o); return NULL; }
        skip_ws(p);
        if (p->p < p->end && *p->p == ':') p->p++;
        jval *v = parse_value(p);
        if (!v) { free(key); json_free(o); return NULL; }
        if (o->n == (int)cap) {
            cap = cap ? cap * 2 : 8;
            o->items = realloc(o->items, cap * sizeof *o->items);
            o->keys = realloc(o->keys, cap * sizeof *o->keys);
        }
        o->keys[o->n] = key;
        o->items[o->n] = v;
        o->n++;
    }
}

static jval *parse_array(P *p)
{
    jval *a = jnew(J_ARR);
    if (!a) return NULL;
    p->p++; /* [ */
    size_t cap = 0;
    for (;;) {
        skip_ws(p);
        if (p->p >= p->end) { json_free(a); return NULL; }
        if (*p->p == ']') { p->p++; return a; }
        if (*p->p == ',') { p->p++; continue; }
        jval *v = parse_value(p);
        if (!v) { json_free(a); return NULL; }
        if (a->n == (int)cap) {
            cap = cap ? cap * 2 : 8;
            a->items = realloc(a->items, cap * sizeof *a->items);
        }
        a->items[a->n++] = v;
    }
}

static jval *parse_value(P *p)
{
    skip_ws(p);
    if (p->p >= p->end) return NULL;
    char c = *p->p;
    if (c == '{') return parse_object(p);
    if (c == '[') return parse_array(p);
    if (c == '"') {
        jval *v = jnew(J_STR);
        if (!v) return NULL;
        v->s = parse_string(p);
        if (!v->s) { json_free(v); return NULL; }
        return v;
    }
    if (c == 't') {
        if (p->end - p->p >= 4 && !strncmp(p->p, "true", 4)) {
            p->p += 4;
            jval *v = jnew(J_BOOL);
            if (v) v->i = 1;
            return v;
        }
        return NULL;
    }
    if (c == 'f') {
        if (p->end - p->p >= 5 && !strncmp(p->p, "false", 5)) {
            p->p += 5;
            return jnew(J_BOOL);
        }
        return NULL;
    }
    if (c == 'n') {
        if (p->end - p->p >= 4 && !strncmp(p->p, "null", 4)) {
            p->p += 4;
            return jnew(J_NULL);
        }
        return NULL;
    }
    /* number */
    char *endp;
    double d = strtod(p->p, &endp);
    if (endp == p->p) return NULL;
    jval *v = jnew(J_DOUBLE);
    if (!v) return NULL;
    v->num = d;
    const char *q = p->p;
    int integral = 1;
    for (; q < endp; q++) {
        if (*q == '.' || *q == 'e' || *q == 'E') { integral = 0; break; }
    }
    if (integral) {
        v->t = J_INT;
        v->i = (long long)d;
    }
    p->p = endp;
    return v;
}

jval *json_parse_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[len] = '\0';
    P p = {buf, buf + len};
    jval *v = parse_value(&p);
    free(buf);
    return v;
}

jval *json_get(const jval *obj, const char *key)
{
    if (!obj || obj->t != J_OBJ) return NULL;
    for (int i = 0; i < obj->n; i++) {
        if (!strcmp(obj->keys[i], key)) return obj->items[i];
    }
    return NULL;
}

double json_get_num(const jval *obj, const char *key, double def)
{
    jval *v = json_get(obj, key);
    if (!v) return def;
    if (v->t == J_INT || v->t == J_DOUBLE) return v->num;
    return def;
}

long long json_get_int(const jval *obj, const char *key, long long def)
{
    jval *v = json_get(obj, key);
    if (!v) return def;
    if (v->t == J_INT) return v->i;
    if (v->t == J_DOUBLE) return (long long)v->num;
    return def;
}

int json_get_bool(const jval *obj, const char *key, int def)
{
    jval *v = json_get(obj, key);
    if (!v) return def;
    if (v->t == J_BOOL) return v->i != 0;
    if (v->t == J_INT) return v->i != 0;
    return def;
}
