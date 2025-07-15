"""
Navigation System Receiver Example
Demonstrates how the navigation system can receive and process object detection data
This would be used by your teammate's SLAM navigation system
"""

import socket
import json
import threading
import time
from typing import Dict, List, Optional
import numpy as np
from collections import deque
import argparse

class NavigationReceiver:
    """
    Example navigation system receiver that processes object detection data
    """
    
    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
        # Data storage
        self.received_objects = deque(maxlen=1000)  # Store last 1000 detections
        self.landmarks = {}  # Current landmarks for navigation
        self.object_history = {}  # Track object history by ID
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'objects_processed': 0,
            'landmarks_created': 0,
            'start_time': time.time()
        }
        
        # Thread for processing
        self.receiver_thread = None
        self.processor_thread = None
        
        # Navigation parameters
        self.landmark_distance_threshold = 1.0  # meters
        self.min_observations = 3  # Minimum observations to create landmark
        self.landmark_timeout = 30.0  # seconds
        
        print(f"🚀 Navigation Receiver initialized")
        print(f"📡 Listening on {host}:{port}")
    
    def start(self):
        """Start the navigation receiver"""
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            print(f"✅ Socket server started on {self.host}:{self.port}")
            
            self.running = True
            
            # Start receiver thread
            self.receiver_thread = threading.Thread(target=self._receiver_worker, daemon=True)
            self.receiver_thread.start()
            
            # Start processor thread
            self.processor_thread = threading.Thread(target=self._processor_worker, daemon=True)
            self.processor_thread.start()
            
            print("🔄 Navigation receiver started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start navigation receiver: {e}")
            return False
    
    def _receiver_worker(self):
        """Worker thread for receiving data"""
        while self.running:
            try:
                # Accept connection
                client_socket, address = self.socket.accept()
                print(f"🔗 Connection from {address}")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"❌ Receiver error: {e}")
                break
    
    def _handle_client(self, client_socket, address):
        """Handle individual client connection"""
        try:
            buffer = ""
            while self.running:
                # Receive data
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Process complete messages (separated by newlines)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self._process_message(line.strip())
                        
        except Exception as e:
            print(f"❌ Client handling error: {e}")
        finally:
            client_socket.close()
            print(f"🔌 Disconnected from {address}")
    
    def _process_message(self, message: str):
        """Process received message"""
        try:
            data = json.loads(message)
            
            # Extract objects from message
            objects = data.get('objects', [])
            timestamp = data.get('timestamp', time.time())
            
            # Add to received objects
            for obj in objects:
                obj['received_timestamp'] = timestamp
                self.received_objects.append(obj)
            
            self.stats['messages_received'] += 1
            self.stats['objects_processed'] += len(objects)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
        except Exception as e:
            print(f"❌ Message processing error: {e}")
    
    def _processor_worker(self):
        """Worker thread for processing received data"""
        while self.running:
            try:
                # Process received objects
                self._process_objects_for_navigation()
                
                # Clean up old landmarks
                self._cleanup_old_landmarks()
                
                # Print statistics periodically
                if self.stats['messages_received'] % 10 == 0 and self.stats['messages_received'] > 0:
                    self._print_statistics()
                
                time.sleep(1.0)  # Process every second
                
            except Exception as e:
                print(f"❌ Processor error: {e}")
    
    def _process_objects_for_navigation(self):
        """Process objects for navigation and SLAM"""
        # Process recent objects
        recent_objects = []
        current_time = time.time()
        
        for obj in self.received_objects:
            if current_time - obj.get('received_timestamp', 0) < 5.0:  # Last 5 seconds
                recent_objects.append(obj)
        
        # Group objects by ID and update landmarks
        object_groups = {}
        for obj in recent_objects:
            obj_id = obj.get('object_id', 0)
            if obj_id not in object_groups:
                object_groups[obj_id] = []
            object_groups[obj_id].append(obj)
        
        # Update landmarks based on grouped objects
        for obj_id, objects in object_groups.items():
            if len(objects) >= self.min_observations:
                self._update_or_create_landmark(obj_id, objects)
    
    def _update_or_create_landmark(self, obj_id: int, objects: List[Dict]):
        """Update or create landmark from object observations"""
        try:
            # Calculate average position
            positions = []
            confidences = []
            class_names = []
            
            for obj in objects:
                if 'position_3d' in obj:
                    positions.append(obj['position_3d'])
                    confidences.append(obj.get('confidence', 0.5))
                    class_names.append(obj.get('class_name', 'unknown'))
            
            if not positions:
                return
            
            # Calculate average position
            avg_position = np.mean(positions, axis=0).tolist()
            avg_confidence = np.mean(confidences)
            most_common_class = max(set(class_names), key=class_names.count)
            
            # Create or update landmark
            landmark = {
                'id': obj_id,
                'type': most_common_class,
                'position': avg_position,
                'confidence': avg_confidence,
                'observations': len(objects),
                'last_seen': time.time(),
                'uncertainty': np.std(positions, axis=0).tolist() if len(positions) > 1 else [0.1, 0.1, 0.1]
            }
            
            self.landmarks[obj_id] = landmark
            
            if obj_id not in self.object_history:
                self.stats['landmarks_created'] += 1
                print(f"🏷️  New landmark created: {most_common_class} at {avg_position}")
            
            # Update history
            self.object_history[obj_id] = {
                'positions': positions[-10:],  # Keep last 10 positions
                'confidences': confidences[-10:],
                'class_names': class_names[-10:]
            }
            
        except Exception as e:
            print(f"❌ Landmark update error: {e}")
    
    def _cleanup_old_landmarks(self):
        """Remove old landmarks that haven't been seen recently"""
        current_time = time.time()
        to_remove = []
        
        for landmark_id, landmark in self.landmarks.items():
            if current_time - landmark['last_seen'] > self.landmark_timeout:
                to_remove.append(landmark_id)
        
        for landmark_id in to_remove:
            del self.landmarks[landmark_id]
            if landmark_id in self.object_history:
                del self.object_history[landmark_id]
    
    def get_navigation_landmarks(self) -> List[Dict]:
        """Get landmarks for navigation system"""
        navigation_landmarks = []
        
        for landmark in self.landmarks.values():
            if (landmark['observations'] >= self.min_observations and 
                landmark['confidence'] > 0.5):
                
                # Convert to navigation format
                nav_landmark = {
                    'id': landmark['id'],
                    'type': landmark['type'],
                    'position': landmark['position'],  # [x, y, z] in meters
                    'confidence': landmark['confidence'],
                    'uncertainty': np.linalg.norm(landmark['uncertainty']),  # Scalar uncertainty
                    'observations': landmark['observations'],
                    'is_static': True  # Assume static for SLAM
                }
                navigation_landmarks.append(nav_landmark)
        
        return navigation_landmarks
    
    def get_obstacle_map(self) -> Dict:
        """Get obstacle map for navigation"""
        obstacles = []
        
        for landmark in self.landmarks.values():
            if landmark['confidence'] > 0.7:  # High confidence obstacles
                obstacle = {
                    'position': landmark['position'],
                    'type': landmark['type'],
                    'radius': self._estimate_object_radius(landmark['type']),
                    'confidence': landmark['confidence']
                }
                obstacles.append(obstacle)
        
        return {
            'obstacles': obstacles,
            'timestamp': time.time(),
            'coordinate_frame': 'camera'
        }
    
    def _estimate_object_radius(self, object_type: str) -> float:
        """Estimate object radius for collision avoidance"""
        radius_map = {
            'person': 0.3,
            'chair': 0.4,
            'table': 0.6,
            'bottle': 0.05,
            'cup': 0.04,
            'laptop': 0.2,
            'book': 0.1,
            'plant': 0.2,
            'monitor': 0.3,
            'default': 0.2
        }
        
        return radius_map.get(object_type.lower(), radius_map['default'])
    
    def _print_statistics(self):
        """Print receiver statistics"""
        runtime = time.time() - self.stats['start_time']
        print(f"\n📊 Navigation Receiver Statistics:")
        print(f"   Messages received: {self.stats['messages_received']}")
        print(f"   Objects processed: {self.stats['objects_processed']}")
        print(f"   Active landmarks: {len(self.landmarks)}")
        print(f"   Landmarks created: {self.stats['landmarks_created']}")
        print(f"   Runtime: {runtime:.1f}s")
        print(f"   Messages/sec: {self.stats['messages_received'] / runtime:.2f}")
    
    def export_landmarks(self, filename: str = "navigation_landmarks.json"):
        """Export landmarks for analysis"""
        try:
            export_data = {
                'timestamp': time.time(),
                'landmarks': list(self.landmarks.values()),
                'statistics': {
                    **self.stats,
                    'runtime': time.time() - self.stats['start_time']
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"💾 Landmarks exported to {filename}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def stop(self):
        """Stop the navigation receiver"""
        print("\n🛑 Stopping navigation receiver...")
        self.running = False
        
        if self.socket:
            self.socket.close()
        
        if self.receiver_thread:
            self.receiver_thread.join(timeout=2.0)
        
        if self.processor_thread:
            self.processor_thread.join(timeout=2.0)
        
        # Export final landmarks
        self.export_landmarks("final_navigation_landmarks.json")
        
        print("✅ Navigation receiver stopped")

def main():
    """Main function for standalone receiver"""
    parser = argparse.ArgumentParser(description='Navigation System Receiver')
    parser.add_argument('--host', type=str, default='localhost', help='Host to listen on')
    parser.add_argument('--port', type=int, default=8888, help='Port to listen on')
    parser.add_argument('--export-interval', type=int, default=60, help='Export interval in seconds')
    
    args = parser.parse_args()
    
    # Create receiver
    receiver = NavigationReceiver(args.host, args.port)
    
    try:
        if receiver.start():
            print("\n🔄 Navigation receiver running...")
            print("Press Ctrl+C to stop")
            
            # Export landmarks periodically
            last_export = time.time()
            
            while receiver.running:
                time.sleep(1.0)
                
                # Periodic export
                if time.time() - last_export > args.export_interval:
                    receiver.export_landmarks(f"landmarks_{int(time.time())}.json")
                    last_export = time.time()
                
                # Show current landmarks
                landmarks = receiver.get_navigation_landmarks()
                if landmarks:
                    print(f"\n🏷️  Current landmarks: {len(landmarks)}")
                    for landmark in landmarks[:5]:  # Show first 5
                        pos = landmark['position']
                        print(f"   {landmark['type']}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        receiver.stop()

if __name__ == "__main__":
    main() 