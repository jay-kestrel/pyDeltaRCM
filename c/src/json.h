#ifndef PYD_JSON_H
#define PYD_JSON_H

#include <stddef.h>

/*
 * Minimal JSON parser for the normalized flat config files.
 *
 * Supports null/bool/int/double/string/array/object. Integers are kept as
 * long long (plus their double value) so integral params like Np_sed stay
 * exact; everything else parses through strtod.
 */
typedef enum { J_NULL, J_BOOL, J_INT, J_DOUBLE, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype         t;
    double        num;      /* value for J_INT and J_DOUBLE */
    long long     i;        /* value for J_INT */
    char         *s;        /* J_STR */
    struct jval **items;    /* J_ARR / J_OBJ element values */
    char        **keys;     /* J_OBJ keys (parallel to items) */
    int           n;
} jval;

jval *json_parse_file(const char *path);
void  json_free(jval *v);

jval *json_get(const jval *obj, const char *key);
double    json_get_num(const jval *obj, const char *key, double def);
long long json_get_int(const jval *obj, const char *key, long long def);
int       json_get_bool(const jval *obj, const char *key, int def);

#endif
