#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ inline float luminance(const glm::vec3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{

    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    if (pathSegment.remainingBounces <= 0) {
        return;
    }
    pathSegment.remainingBounces -= 1;
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (m.emittance > 0.0f) {
        pathSegment.color *= (m.color * m.emittance);
        pathSegment.remainingBounces = 0;
        return;
    }
    if (m.hasRefractive) {
        //pass
        return;
    }
    float pDiff = fmaxf(luminance(m.color), 0.0f), pSpec = (m.hasReflective > 0.0f) ? fmaxf(luminance(m.specular.color), 0.0f) : 0.0f;

    pDiff = pDiff / (pDiff + pSpec);
    if (u01(rng) < pDiff) {
        // we diffuse
        glm::vec3 newDir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + EPSILON * normal;
        pathSegment.ray.direction = newDir;
        pathSegment.color *= m.color / fmaxf(pDiff, 1e-6f);
    }
    else {
        // we reflect
        glm::vec3 newDir = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect + EPSILON * normal;
        pathSegment.ray.direction = newDir;
        pathSegment.color *= m.specular.color / fmaxf(1.0f - pDiff, 1e-6f);
    }
}
