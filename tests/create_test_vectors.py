import random
from py_ecc.bls12_381 import G1, add, multiply, FQ

MODULUS = int("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", 16)

def sample_random_affine_point():
    
    random_scalar = random.randint(1, MODULUS - 1)
    random_point = multiply(G1, random_scalar)

    return random_point

def read_points_from_file(lines, filename):

    points = []
    try:
        with open(filename, 'r') as file:
            for i in range(0, 2 * lines, 2):

                # Remove any newline characters and cast to integer
                x = int(next(file).strip())
                y = int(next(file).strip())

                point = (FQ(x), FQ(y))
                points.append(point)

    except FileNotFoundError:
        print(f"The file {filename} does not exist.")
    except ValueError:
        print("There was an error converting one of the lines to an integer.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return points


def generate_test_vectors(filename=None):

    points = []
    if filename is None:
        # Sample random affine points
        for _ in range(64 * 1024):
            point = sample_random_affine_point()
            points.append(point)

        # Write points to a file
        with open('bls12-381_points.txt', 'w') as f:
            for x, y in points:
                f.write(f'{x.n}\n{y.n}\n')
    else:
        points = read_points_from_file(64 * 1024, filename)

    #---------------------------------------------------------
    # Generate expected results for the test vectors
    #---------------------------------------------------------

    ### test_add_points.txt ###
    test_add_points = []
    for i in range(0, len(points), 2):
        sum = add(points[i], points[i+1])
        test_add_points.append(sum)

    with open('test_add_points.txt', 'w') as f:
        for x, y in test_add_points:
            f.write(f'{x.n}\n{y.n}\n')

    ### test_accumulate_points.txt ###
    test_accumulate_points = []
    points_per_thread = 32
    for j in range(0 , len(points), points_per_thread):
        acc = points[j]
        for i in range(1, points_per_thread):
            acc = add(acc, points[i + j])

        test_accumulate_points.append(acc)

    with open('test_accumulate_points.txt', 'w') as f:
        for x, y in test_accumulate_points:
            f.write(f'{x.n}\n{y.n}\n')

    ### test_reduce_points.txt ###
    result = test_accumulate_points[0]
    for point in test_accumulate_points[1:]:
        result = add(result, point)

    with open('test_reduce_points.txt', 'w') as f:
        f.write(f'{result[0].n}\n{result[1].n}\n')

if __name__ == '__main__':
    # Don't specify filename if you want to generate random points
    generate_test_vectors(filename='bls12-381_points.txt')