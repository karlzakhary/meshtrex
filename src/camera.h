#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

class Camera {
public:
    Camera() {
        position_ = glm::vec3(0.0f, 0.0f, 5.0f);
        front_ = glm::vec3(0.0f, 0.0f, -1.0f);
        up_ = glm::vec3(0.0f, 1.0f, 0.0f);
        worldUp_ = up_;
        
        yaw_ = -90.0f;  // Looking along -Z
        pitch_ = 0.0f;
        
        movementSpeed_ = 2.5f;
        mouseSensitivity_ = 0.1f;
        fov_ = 45.0f;
        nearPlane_ = 0.1f;
        farPlane_ = 1000.0f;
        
        changed_ = true;
        updateCameraVectors();
    }
    
    // Get matrices
    glm::mat4 getViewMatrix() const {
        return glm::lookAt(position_, position_ + front_, up_);
    }
    
    glm::mat4 getProjectionMatrix(float aspectRatio) const {
        return glm::perspective(glm::radians(fov_), aspectRatio, nearPlane_, farPlane_);
    }
    
    // Position and orientation
    void setPosition(const glm::vec3& position) {
        position_ = position;
        changed_ = true;
    }
    
    void lookAt(const glm::vec3& target) {
        front_ = glm::normalize(target - position_);
        updateEulerAngles();
        changed_ = true;
    }
    
    // Movement
    void moveForward(float deltaTime) {
        position_ += front_ * movementSpeed_ * deltaTime;
        changed_ = true;
    }
    
    void moveBackward(float deltaTime) {
        position_ -= front_ * movementSpeed_ * deltaTime;
        changed_ = true;
    }
    
    void moveLeft(float deltaTime) {
        position_ -= right_ * movementSpeed_ * deltaTime;
        changed_ = true;
    }
    
    void moveRight(float deltaTime) {
        position_ += right_ * movementSpeed_ * deltaTime;
        changed_ = true;
    }
    
    void moveUp(float deltaTime) {
        position_ += worldUp_ * movementSpeed_ * deltaTime;
        changed_ = true;
    }
    
    void moveDown(float deltaTime) {
        position_ -= worldUp_ * movementSpeed_ * deltaTime;
        changed_ = true;
    }
    
    // Rotation
    void rotate(float yawOffset, float pitchOffset) {
        yaw_ += yawOffset * mouseSensitivity_;
        pitch_ += pitchOffset * mouseSensitivity_;
        
        // Constrain pitch
        if (pitch_ > 89.0f) pitch_ = 89.0f;
        if (pitch_ < -89.0f) pitch_ = -89.0f;
        
        updateCameraVectors();
        changed_ = true;
    }
    
    // Zoom
    void zoom(float offset) {
        fov_ -= offset;
        if (fov_ < 1.0f) fov_ = 1.0f;
        if (fov_ > 90.0f) fov_ = 90.0f;
        changed_ = true;
    }
    
    // Properties
    void setMovementSpeed(float speed) { movementSpeed_ = speed; }
    void setMouseSensitivity(float sensitivity) { mouseSensitivity_ = sensitivity; }
    void setFieldOfView(float fov) { fov_ = fov; changed_ = true; }
    void setNearPlane(float near) { nearPlane_ = near; changed_ = true; }
    void setFarPlane(float far) { farPlane_ = far; changed_ = true; }
    
    // Getters
    glm::vec3 getPosition() const { return position_; }
    glm::vec3 getFront() const { return front_; }
    glm::vec3 getUp() const { return up_; }
    glm::vec3 getRight() const { return right_; }
    float getYaw() const { return yaw_; }
    float getPitch() const { return pitch_; }
    float getFieldOfView() const { return fov_; }
    float getNearPlane() const { return nearPlane_; }
    float getFarPlane() const { return farPlane_; }
    
    // Change tracking
    bool hasChanged() const { return changed_; }
    void resetChanged() { changed_ = false; }
    
    // Frustum extraction for culling
    void extractFrustumPlanes(const glm::mat4& viewProj, glm::vec4 frustumPlanes[6]) const {
        // Left plane
        frustumPlanes[0] = glm::vec4(
            viewProj[0][3] + viewProj[0][0],
            viewProj[1][3] + viewProj[1][0],
            viewProj[2][3] + viewProj[2][0],
            viewProj[3][3] + viewProj[3][0]
        );
        
        // Right plane
        frustumPlanes[1] = glm::vec4(
            viewProj[0][3] - viewProj[0][0],
            viewProj[1][3] - viewProj[1][0],
            viewProj[2][3] - viewProj[2][0],
            viewProj[3][3] - viewProj[3][0]
        );
        
        // Bottom plane
        frustumPlanes[2] = glm::vec4(
            viewProj[0][3] + viewProj[0][1],
            viewProj[1][3] + viewProj[1][1],
            viewProj[2][3] + viewProj[2][1],
            viewProj[3][3] + viewProj[3][1]
        );
        
        // Top plane
        frustumPlanes[3] = glm::vec4(
            viewProj[0][3] - viewProj[0][1],
            viewProj[1][3] - viewProj[1][1],
            viewProj[2][3] - viewProj[2][1],
            viewProj[3][3] - viewProj[3][1]
        );
        
        // Near plane
        frustumPlanes[4] = glm::vec4(
            viewProj[0][3] + viewProj[0][2],
            viewProj[1][3] + viewProj[1][2],
            viewProj[2][3] + viewProj[2][2],
            viewProj[3][3] + viewProj[3][2]
        );
        
        // Far plane
        frustumPlanes[5] = glm::vec4(
            viewProj[0][3] - viewProj[0][2],
            viewProj[1][3] - viewProj[1][2],
            viewProj[2][3] - viewProj[2][2],
            viewProj[3][3] - viewProj[3][2]
        );
        
        // Normalize planes
        for (int i = 0; i < 6; i++) {
            float length = glm::length(glm::vec3(frustumPlanes[i]));
            frustumPlanes[i] /= length;
        }
    }

private:
    // Camera attributes
    glm::vec3 position_;
    glm::vec3 front_;
    glm::vec3 up_;
    glm::vec3 right_;
    glm::vec3 worldUp_;
    
    // Euler angles
    float yaw_;
    float pitch_;
    
    // Camera options
    float movementSpeed_;
    float mouseSensitivity_;
    float fov_;
    float nearPlane_;
    float farPlane_;
    
    // State tracking
    bool changed_;
    
    void updateCameraVectors() {
        // Calculate new front vector
        glm::vec3 front;
        front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front.y = sin(glm::radians(pitch_));
        front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front_ = glm::normalize(front);
        
        // Recalculate right and up vectors
        right_ = glm::normalize(glm::cross(front_, worldUp_));
        up_ = glm::normalize(glm::cross(right_, front_));
    }
    
    void updateEulerAngles() {
        // Calculate yaw and pitch from front vector
        yaw_ = glm::degrees(atan2(front_.z, front_.x));
        pitch_ = glm::degrees(asin(front_.y));
    }
};