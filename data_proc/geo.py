import math

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
        self.len = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __str__(self):
        return "{} {} {}".format(self.x, self.y, self.z)

    def normalise(self):
        try:
            self.x /= self.len
            self.y /= self.len
            self.z /= self.len
        except ZeroDivisionError:
            print('Warning: Length of the vector seems to be zero.')

    def dot_product(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def inv_dot_product(self, v):
        return math.acos(self.dot_product(v))

    def cross_product(self, v):
        return Vector(self.y * v.z - self.z * v.y, self.z * v.x - self.x * v.z, self.x * v.y - self.y * v.x)

    def rotate_vector(self, q):
        x = (1 - 2 * q.y * q.y - 2 * q.z * q.z) * self.x +\
                2 * (q.x * q.y + q.w * q.z) * self.y +\
                2 * (q.x * q.z - q.w * q.y) * self.z
        y = 2 * (q.x * q.y - q.w * q.z) * self.x +\
                (1 - 2 * q.x * q.x - 2 * q.z * q.z) * self.y +\
                2 * (q.y * q.z + q.w * q.x) * self.z
        z = 2 * (q.x * q.z + q.w * q.y) * self.x +\
                2 * (q.y * q.z - q.w * q.x) * self.y +\
                (1 - 2 * q.x * q.x - 2 * q.y * q.y) * self.z
        # print('Pre-rotate: {} {} {}'.format(x, y, z))

        return Vector(x, y, z)

    def get_len(self):
        return self.len

class Quaternion:
    def __init__(self, *args, **kwargs):
        if kwargs.get('x') != None:
            self.x = kwargs.get('x')
            self.y = kwargs.get('y')
            self.z = kwargs.get('z')
            self.w = kwargs.get('w')

        elif kwargs.get('vector') != None:
            v = kwargs.get('vector')
            d = kwargs.get('angle')

            self.x = v.x * math.sin(d / 2.0)
            self.y = v.y * math.sin(d / 2.0)
            self.z = v.z * math.sin(d / 2.0)
            self.w = math.cos(d / 2.0)

        # elif kwargs.get('roll') != None:
            # roll = kwargs.get('roll')
            # pitch = kwargs.get('pitch')
            # yaw = kwargs.get('yaw')

            # t0 = math.cos(yaw * 0.5)
            # t1 = math.sin(yaw * 0.5)
            # t2 = math.cos(roll * 0.5)
            # t3 = math.sin(roll * 0.5)
            # t4 = math.cos(pitch * 0.5)
            # t5 = math.sin(pitch * 0.5)

            # self.w = t0 * t2 * t4 + t1 * t3 * t5
            # self.x = t0 * t3 * t4 - t1 * t2 * t5
            # self.y = t0 * t2 * t5 + t1 * t3 * t4
            # self.z = t1 * t2 * t4 - t0 * t3 * t5

        self.len = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2 + self.w ** 2)

    def __str__(self):
        return "{} {} {} {}".format(self.x, self.y, self.z, self.w)

    def normalise(self):
        try:
            self.x /= self.len
            self.y /= self.len
            self.z /= self.len
            self.w /= self.len
        except ZeroDivisionError:
            print('Warning: The length of the vector seems to be zero')

    def multiply(self, q):
        # Rotate quaternion by q
        return Quaternion(x=self.w * q.x + self.x * q.w - self.y * q.z + self.z * q.y,
                          y=self.w * q.y + self.x * q.z + self.y * q.w - self.z * q.x,
                          z=self.w * q.z - self.x * q.y + self.y * q.x + self.z * q.w,
                          w=self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z)
