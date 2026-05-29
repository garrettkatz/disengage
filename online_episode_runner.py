import zmq
import numpy as np
# import json
# import pybullet as p
# import pybullet_data
# import time

# 1. Setup ZeroMQ Client
context = zmq.Context()
socket = context.socket(zmq.REQ)
# Connect to Jetson Nano's USB-C IP
jetson_ip = "192.168.55.1" 
socket.connect(f"tcp://{jetson_ip}:5555")

# Send state to Jetson
state = np.random.rand(12)
img = np.random.randint(255, size=(48,64), dtype=np.uint8)
print(state)
print(img[:2,:3])
# socket.send(state, copy=False)
socket.send_multipart([state.tobytes(), img.tobytes()], copy=False)

# # 2. Setup PyBullet Environment
# physicsClient = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.8)
# planeId = p.loadURDF("plane.urdf")
# # Example object: Replace with your actual robot/agent URDF
# robotId = p.loadURDF("r2d2.urdf", [0, 0, 1])

# print("Starting simulation loop...")

# # 3. Episode Loop
# for episode in range(10):
#     # Reset or initialize your environment state here
#     print(f"Starting Episode {episode}")
    
#     for step in range(1000):
#         # Gather your state observation (Example: 8 mock floating point values)
#         # In reality, extract this using p.getBasePositionAndOrientation(robotId), etc.
#         state_observation = [0.1, -0.2, 0.5, 1.2, -0.4, 0.0, 0.8, -0.1]
        
        # # Send state to Jetson
        # socket.send_json({'state': state_observation})
        
#         # Wait for action reply (blocks until Jetson responds)
#         reply = socket.recv_json()
#         action = reply['action']
        
#         # Apply the action to your PyBullet agent
#         # Example: p.setJointMotorControlArray(robotId, ...)
        
#         # Step the simulator physics forward
#         p.stepSimulation()
#         time.sleep(1./240.) 

# p.disconnect()

