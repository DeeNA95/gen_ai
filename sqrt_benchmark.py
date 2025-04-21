import timeit
import math

# Number of iterations
number = 100000000
# Value to calculate square root for
value = 102334435594949953995355432312345.6789453434923848280248240244343535222

# Time math.sqrt()
sqrt_time = timeit.timeit(lambda: math.sqrt(value), number=number)

# Time **0.5
pow_time = timeit.timeit(lambda: value**0.5, number=number)

print(f"Benchmarking over {number} iterations for value {value}:")
print(f"math.sqrt(): {sqrt_time:.6f} seconds")
print(f"**0.5:       {pow_time:.6f} seconds")

if sqrt_time < pow_time:
    print("\nmath.sqrt() is faster.")
    print(f"Difference: {pow_time - sqrt_time:.6f} seconds")
    print(f"Ratio (pow_time / sqrt_time): {pow_time / sqrt_time:.2f}x faster")
else:
    print("\n**0.5 is faster.")
    print(f"Difference: {sqrt_time - pow_time:.6f} seconds")
    print(f"Ratio (sqrt_time / pow_time): {sqrt_time / pow_time:.2f}x faster")
