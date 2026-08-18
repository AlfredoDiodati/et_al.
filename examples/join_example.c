#include <stdio.h>
#include "frame/join.h"

/* Two small, realistic tables: orders placed by customer_id, and a
   customers table with one row missing an order and one id ("c4")
   present only on the customers side - just enough to make INNER,
   LEFT, FULL and the right-join-via-swap idiom all produce visibly
   different results. */
static DataFrame make_orders(void) {
    DataFrame df = df_new(5);
    const char *customer_id[5] = { "c1", "c2", "c1", "c3", "c3" };
    df_add_string_col(&df, "customer_id", customer_id);
    Vec amount = mat_lit(5, 1, 19.99f, 42.50f, 8.00f, 100.00f, 15.25f);
    df_add_numeric_col(&df, "amount", amount); mat_free(amount);
    return df;
}

static DataFrame make_customers(void) {
    DataFrame df = df_new(4);
    const char *customer_id[4] = { "c1", "c2", "c3", "c4" };
    df_add_string_col(&df, "customer_id", customer_id);
    const char *name[4] = { "Alice", "Bob", "Carol", "Dave" };
    df_add_string_col(&df, "name", name);
    return df;
}

int main(void) {
    DataFrame orders = make_orders();
    DataFrame customers = make_customers();

    printf("orders:\n"); df_print(&orders);
    printf("customers:\n"); df_print(&customers);

    /* INNER: only orders whose customer_id also has a customers row.
       Every order here matches, since orders never reference "c4" and
       every customer_id in orders exists in customers. */
    DataFrame inner = df_join(&orders, &customers, "customer_id", JOIN_INNER);
    printf("INNER join (orders, customers):\n"); df_print(&inner);

    /* LEFT (= left outer): every order survives regardless of match.
       Same result as INNER here, since every order does match - the
       difference only shows up when a left row has no counterpart. */
    DataFrame left = df_join(&orders, &customers, "customer_id", JOIN_LEFT);
    printf("LEFT join (orders, customers):\n"); df_print(&left);

    /* FULL: every order AND every customer, including "c4" (Dave, who
       has never ordered) - his row's "amount" comes back NaN, the
       genuine missing-value marker JOIN_LEFT/JOIN_FULL introduce for
       numeric columns (see docs/JOIN_DOCUMENTATION.md, "What the output
       looks like"). */
    DataFrame full = df_join(&orders, &customers, "customer_id", JOIN_FULL);
    printf("FULL join (orders, customers):\n"); df_print(&full);

    /* RIGHT join has no separate entry point - it is a LEFT join with
       the two DataFrames swapped, "every customer, matched orders":
       df_join(&customers, &orders, on, JOIN_LEFT). Dave (no orders)
       still appears once, with amount NaN; Carol (two orders) appears
       twice, ordinary duplicate-key fan-out. */
    DataFrame right = df_join(&customers, &orders, "customer_id", JOIN_LEFT);
    printf("RIGHT join (customers, orders) via df_join(customers, orders, on, JOIN_LEFT):\n");
    df_print(&right);

    df_free(&orders); df_free(&customers);
    df_free(&inner); df_free(&left); df_free(&full); df_free(&right);
    return 0;
}
