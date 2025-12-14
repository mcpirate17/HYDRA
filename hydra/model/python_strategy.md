•  Use Built-in Data Structures and Functions:
	•  Leverage Python’s optimized built-ins like lists, sets, and dictionaries instead of custom implementations.
	•  Example: Use set for membership testing (O(1)) instead of lists (O(n)).
	•  Auditor Check: Flag loops or custom logic that could be replaced with built-in methods (e.g., in with lists vs. sets).
•  Minimize Loop Overhead:
	•  Replace explicit loops with list comprehensions, generator expressions, or built-in functions like map(), filter(), and sum() when appropriate.
	•  Example: [x**2 for x in range(n)] is faster than a for loop appending to a list.
	•  Auditor Check: Detect nested loops or loops that can be vectorized/comprehended.
•  Leverage NumPy for Numerical Operations:
	•  Use NumPy arrays for numerical computations instead of Python lists to exploit vectorization and C-based optimizations.
	•  Example: np.array(x) + np.array(y) is faster than [x[i] + y[i] for i in range(len(x))].
	•  Auditor Check: Flag numerical loops or list-based math that could use NumPy.
•  Avoid Unnecessary Object Copies:
	•  Use in-place operations (e.g., +=, list.extend()) to avoid creating new objects.
	•  Example: lst.extend(other) is more efficient than lst = lst + other.
	•  Auditor Check: Identify concatenation in loops or non-in-place operations.
•  Use Generators for Memory Efficiency:
	•  Replace lists with generators for large datasets to reduce memory usage, especially in iterations.
	•  Example: (x**2 for x in range(n)) instead of [x**2 for x in range(n)] for one-time use.
	•  Auditor Check: Flag large list comprehensions in contexts where results are iterated once.
•  Profile and Optimize Hotspots:
	•  Encourage use of profiling tools like cProfile or line_profiler to identify bottlenecks.
	•  Optimize only the critical sections of code (Pareto principle: 20% of code uses 80% of time).
	•  Auditor Check: Warn about unprofiled code in performance-critical sections (e.g., tight loops).
•  Use Multithreading or Multiprocessing for Parallelism:
	•  Use threading for I/O-bound tasks and multiprocessing for CPU-bound tasks to bypass the Global Interpreter Lock (GIL).
	•  Example: Pool.map() for parallelizing independent computations.
	•  Auditor Check: Detect CPU-intensive loops that could be parallelized.
•  Optimize String Operations:
	•  Use ''.join(list_of_strings) instead of string concatenation in loops (+ creates new strings).
	•  Example: ''.join([str(i) for i in range(n)]) vs. s += str(i) in a loop.
	•  Auditor Check: Flag string concatenation in loops.
•  Cache Expensive Computations:
	•  Use functools.lru_cache for memoizing results of expensive function calls with repeatable inputs.
	•  Example: @lru_cache on recursive or frequently called functions like Fibonacci.
	•  Auditor Check: Identify recursive or repetitive function calls without caching.
•  Use Efficient Libraries for Specific Tasks:
	•  Replace standard library implementations with optimized libraries (e.g., pandas for data manipulation, numba for JIT compilation).
	•  Example: Use numba.jit to compile numerical functions to machine code.
	•  Auditor Check: Suggest library alternatives for common tasks (e.g., regex vs. re module).
•  Avoid Global Variables:
	•  Minimize use of global variables, as they slow down name resolution and make code harder to optimize.
	•  Example: Pass variables as function arguments instead of accessing globals.
	•  Auditor Check: Flag global variable usage in performance-critical functions.
•  Preallocate Data Structures:
	•  Initialize lists or arrays with known sizes to avoid dynamic resizing.
	•  Example: lst = [0] * n vs. appending n times in a loop.
	•  Auditor Check: Detect dynamic list growth in loops.
•  Use Slots for Memory-Efficient Classes:
	•  Define __slots__ in classes to reduce memory overhead for objects with fixed attributes.
	•  Example: class Point: __slots__ = ('x', 'y').
	•  Auditor Check: Suggest __slots__ for classes with many instances.
•  Compile with Cython or PyPy for Speed:
	•  Use Cython to compile Python code to C for performance-critical modules, or run with PyPy for JIT compilation.
	•  Example: Convert bottlenecks to Cython with static typing.
	•  Auditor Check: Flag slow pure-Python code that could benefit from compilation.
•  Minimize Attribute Lookups:
	•  Cache frequently accessed attributes or methods in local variables to reduce lookup overhead.
	•  Example: local_len = len; local_len(lst) vs. len(lst) in a loop.
	•  Auditor Check: Identify repeated attribute access in loops.
•  Use Appropriate Data Structures:
	•  Choose data structures based on operation complexity (e.g., collections.deque for queues, heapq for priority queues).
	•  Example: Use deque for O(1) appends/pops from both ends vs. lists (O(n) for left-side ops).
	•  Auditor Check: Warn about inefficient data structure usage (e.g., lists as queues).
•  Avoid Overusing Exceptions for Control Flow:
	•  Use conditionals instead of try-except blocks for expected cases, as exceptions are slow.
	•  Example: Check if key in dict instead of try: dict[key].
	•  Auditor Check: Flag try-except in loops or non-error cases.
•  Optimize I/O Operations:
	•  Buffer I/O operations with larger reads/writes and use io.BufferedReader or io.BufferedWriter.
	•  Example: Read files in chunks (file.read(8192)) vs. line-by-line for large files.
	•  Auditor Check: Detect inefficient I/O patterns (e.g., small reads in loops).
•  Use Static Typing with Mypy or Type Hints:
	•  Add type hints to enable potential optimizations with tools like Cython or to catch inefficiencies early.
	•  Example: def add(x: int, y: int) -> int: return x + y.
	•  Auditor Check: Suggest type hints for functions in performance-critical code.
•  Keep Code Simple and Readable:
	•  Balance optimization with maintainability; overly complex code can lead to inefficiencies in development and debugging.
	•  Auditor Check: Warn about overly convoluted optimizations that obscure intent.