A pointer is a variable that stores the memory address of another variable.

Key points:
1.  **Declaration:** `int* ptr;` creates a pointer to the `int` type.
2.  **Operators:**
    *   `&` (address-of) — returns the address of a variable (e.g., `ptr = &a;`).
    *   `*` (dereferencing) — accesses the value at an address (e.g., `x = *ptr;`).
3.  **Applications:** Used for dynamic memory allocation (`new`/`delete`), array manipulation, passing parameters by reference, and creating complex data structures. They allow direct memory manipulation, providing flexibility but requiring careful handling.