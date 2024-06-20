import numpy as np
from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import stratified_resample
from filterpy.monte_carlo import residual_resample
from filterpy.monte_carlo import multinomial_resample


class ParticleFilter:
    def __init__(self, num_particles, dim_x):
        self.num_particles = num_particles
        self.dim_x = dim_x
        self.particles = np.zeros((num_particles, dim_x))
        self.weights = np.ones(num_particles) / num_particles

    def initialize(self, mean, cov):
        self.particles = np.random.multivariate_normal(mean, cov, self.num_particles)

    def predict(self, u, std):
        noise = np.random.randn(self.num_particles, self.dim_x) * std
        self.particles += u + noise

    def update(self, z, R, hx, resample_method=systematic_resample):
        zs = np.array([hx(p) for p in self.particles])
        distances = np.linalg.norm(zs - z, axis=1)
        self.weights = np.exp(-0.5 * distances ** 2 / R)
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

        if self.neff() < self.num_particles / 2:
            self.resample(resample_method)

    def resample(self, resample_method):
        indices = resample_method(self.weights)
        self.particles[:] = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)


# Example Usage
num_particles = 1000
dim_x = 7
pf = ParticleFilter(num_particles, dim_x)

# Initial state estimate and covariance
mean = np.zeros(dim_x)
cov = np.eye(dim_x) * 0.1
pf.initialize(mean, cov)

# Process noise standard deviation
std = 0.01

# Example IMU measurements
imu_accel = np.random.randn(3)
imu_gyro = np.random.randn(3)
z = np.hstack((imu_accel, imu_gyro))

# Prediction step
u = np.zeros(dim_x)  # Example: Control input, typically zero for simple prediction
pf.predict(u, std)


# Measurement function
def hx(x):
    return np.array([x[0], x[1], x[2], x[3], x[4], x[5]])


# Measurement noise covariance
R = 0.1

# Update step
pf.update(z, R, hx)

# Estimated state
estimated_state = pf.estimate()
print("Estimated State:", estimated_state)
