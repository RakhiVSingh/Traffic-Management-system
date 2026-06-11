import cv2
import numpy as np
import time
import threading
import queue
import random
import logging
import os
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("traffic_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SmartTrafficSystem")


class PedestrianDetector:
    """Detects pedestrians using HOG descriptor and SVM classifier"""

    def __init__(self):
        # Initialize HOG descriptor for pedestrian detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info("Pedestrian detector initialized")

    def detect(self, frame):
        """
        Detect pedestrians in the given frame

        Args:
            frame: Image frame from the camera

        Returns:
            boxes: Bounding boxes of detected pedestrians
            count: Number of pedestrians detected
        """
        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Detect pedestrians
        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )

        return boxes, len(boxes)


class VehicleDetector:
    """Detects vehicles using Haarcascades"""

    def __init__(self):
        # Load the cascade classifiers
        try:
            self.car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
            if self.car_cascade.empty():
                logger.error("Car cascade file not found or invalid")
                # Use a simulated detector if the cascade file is not found
                self.use_simulated = True
            else:
                self.use_simulated = False
        except:
            logger.error("Error loading car cascade file")
            self.use_simulated = True

        logger.info(
            f"Vehicle detector initialized (Simulated mode: {self.use_simulated})")

    def detect(self, frame):
        """
        Detect vehicles in the given frame

        Args:
            frame: Image frame from the camera

        Returns:
            boxes: Bounding boxes of detected vehicles
            count: Number of vehicles detected
        """
        if self.use_simulated:
            # Simulate vehicle detection with random values for testing
            count = random.randint(0, 10)
            boxes = np.array([[random.randint(0, 500), random.randint(
                0, 300), 100, 100] for _ in range(count)])
            return boxes, count

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect vehicles
        cars = self.car_cascade.detectMultiScale(gray, 1.1, 3)

        return cars, len(cars)


class TrafficLight:
    """Represents a traffic light at an intersection"""

    def __init__(self, id):
        """
        Initialize a traffic light

        Args:
            id: Identifier for the traffic light
        """
        self.id = id
        self.current_state = "red"  # Start with red for safety
        logger.info(f"Traffic light {id} initialized in red state")


class IntersectionController:
    """Controls a traffic light intersection with coordinated lights"""

    def __init__(self):
        # Initialize traffic lights
        self.traffic_lights = {
            "north": TrafficLight("north"),
            "south": TrafficLight("south"),
            "east": TrafficLight("east"),
            "west": TrafficLight("west")
        }

        # Default phase timings (in seconds)
        self.default_timings = {
            "ns_green": 30,
            "ns_yellow": 5,
            "ew_green": 30,
            "ew_yellow": 5
        }

        # Start with default timings
        self.adaptive_timings = self.default_timings.copy()

        # Initialize with phase 0: North/South green, East/West red
        self.current_phase = 0
        self.phase_remaining_time = self.adaptive_timings["ns_green"]

        # Set initial states
        self._update_traffic_light_states()

        logger.info("Intersection controller initialized")

    def update(self):
        """Update the intersection state and remaining time"""
        self.phase_remaining_time -= 1

        # Check if phase should change
        if self.phase_remaining_time <= 0:
            self._advance_to_next_phase()

        # Update traffic light states based on current phase
        self._update_traffic_light_states()

    def _advance_to_next_phase(self):
        """Advance to the next traffic light phase"""
        # Phase cycle:
        # 0: N/S green, E/W red
        # 1: N/S yellow, E/W red
        # 2: N/S red, E/W green
        # 3: N/S red, E/W yellow
        self.current_phase = (self.current_phase + 1) % 4

        # Set the remaining time based on the new phase
        if self.current_phase == 0:  # N/S green, E/W red
            self.phase_remaining_time = self.adaptive_timings["ns_green"]
            logger.info(
                f"Changed to phase 0: N/S green for {self.phase_remaining_time}s")
        elif self.current_phase == 1:  # N/S yellow, E/W red
            self.phase_remaining_time = self.adaptive_timings["ns_yellow"]
            logger.info(
                f"Changed to phase 1: N/S yellow for {self.phase_remaining_time}s")
        elif self.current_phase == 2:  # N/S red, E/W green
            self.phase_remaining_time = self.adaptive_timings["ew_green"]
            logger.info(
                f"Changed to phase 2: E/W green for {self.phase_remaining_time}s")
        elif self.current_phase == 3:  # N/S red, E/W yellow
            self.phase_remaining_time = self.adaptive_timings["ew_yellow"]
            logger.info(
                f"Changed to phase 3: E/W yellow for {self.phase_remaining_time}s")

    def _update_traffic_light_states(self):
        """Update the states of all traffic lights based on the current phase"""
        if self.current_phase == 0:  # N/S green, E/W red
            self.traffic_lights["north"].current_state = "green"
            self.traffic_lights["south"].current_state = "green"
            self.traffic_lights["east"].current_state = "red"
            self.traffic_lights["west"].current_state = "red"
        elif self.current_phase == 1:  # N/S yellow, E/W red
            self.traffic_lights["north"].current_state = "yellow"
            self.traffic_lights["south"].current_state = "yellow"
            self.traffic_lights["east"].current_state = "red"
            self.traffic_lights["west"].current_state = "red"
        elif self.current_phase == 2:  # N/S red, E/W green
            self.traffic_lights["north"].current_state = "red"
            self.traffic_lights["south"].current_state = "red"
            self.traffic_lights["east"].current_state = "green"
            self.traffic_lights["west"].current_state = "green"
        elif self.current_phase == 3:  # N/S red, E/W yellow
            self.traffic_lights["north"].current_state = "red"
            self.traffic_lights["south"].current_state = "red"
            self.traffic_lights["east"].current_state = "yellow"
            self.traffic_lights["west"].current_state = "yellow"

    def adjust_timing(self, traffic_data):
        """
        Adjust timing based on traffic data

        Args:
            traffic_data: Dict containing vehicle and pedestrian counts for each direction
        """
        # Get total counts in each direction
        ns_vehicle_count = traffic_data.get("north", {}).get(
            "vehicles", 0) + traffic_data.get("south", {}).get("vehicles", 0)
        ew_vehicle_count = traffic_data.get("east", {}).get(
            "vehicles", 0) + traffic_data.get("west", {}).get("vehicles", 0)

        ns_pedestrian_count = traffic_data.get("north", {}).get(
            "pedestrians", 0) + traffic_data.get("south", {}).get("pedestrians", 0)
        ew_pedestrian_count = traffic_data.get("east", {}).get(
            "pedestrians", 0) + traffic_data.get("west", {}).get("pedestrians", 0)

        # Adjust green times based on vehicle counts
        if ns_vehicle_count > ew_vehicle_count * 1.5:
            # Much more N/S traffic - increase N/S green time
            self.adaptive_timings["ns_green"] = min(
                60, self.default_timings["ns_green"] + 10)
            self.adaptive_timings["ew_green"] = max(
                15, self.default_timings["ew_green"] - 5)
        elif ew_vehicle_count > ns_vehicle_count * 1.5:
            # Much more E/W traffic - increase E/W green time
            self.adaptive_timings["ew_green"] = min(
                60, self.default_timings["ew_green"] + 10)
            self.adaptive_timings["ns_green"] = max(
                15, self.default_timings["ns_green"] - 5)
        else:
            # Traffic is relatively balanced - use default timings
            self.adaptive_timings["ns_green"] = self.default_timings["ns_green"]
            self.adaptive_timings["ew_green"] = self.default_timings["ew_green"]

        # Adjust based on pedestrian counts
        if ns_pedestrian_count > 5:
            # Increase crossing time for high pedestrian counts
            self.adaptive_timings["ew_green"] = min(
                60, self.adaptive_timings["ew_green"] + 5)

        if ew_pedestrian_count > 5:
            # Increase crossing time for high pedestrian counts
            self.adaptive_timings["ns_green"] = min(
                60, self.adaptive_timings["ns_green"] + 5)

        logger.info(
            f"Adjusted timings: NS green={self.adaptive_timings['ns_green']}s, EW green={self.adaptive_timings['ew_green']}s")

    def adjust_for_emergency(self, direction):
        """
        Adjust lights for emergency vehicles

        Args:
            direction: Direction of emergency vehicle approach
        """
        # Immediately switch to green for the emergency vehicle direction
        if direction in ["north", "south"]:
            if self.current_phase in [2, 3]:  # If E/W has priority
                # Switch to N/S green
                self.current_phase = 0
                self.phase_remaining_time = max(
                    30, self.adaptive_timings["ns_green"])
                logger.info(
                    f"Emergency override: Switching to N/S green for {self.phase_remaining_time}s")
        elif direction in ["east", "west"]:
            if self.current_phase in [0, 1]:  # If N/S has priority
                # Switch to E/W green
                self.current_phase = 2
                self.phase_remaining_time = max(
                    30, self.adaptive_timings["ew_green"])
                logger.info(
                    f"Emergency override: Switching to E/W green for {self.phase_remaining_time}s")

        # Update traffic light states
        self._update_traffic_light_states()

    def apply_environmental_adjustments(self, sensor_data):
        """
        Apply adjustments based on environmental conditions

        Args:
            sensor_data: Dict containing environmental sensor readings
        """
        if sensor_data.get("is_raining", False):
            # Increase all green times during rain to account for slower traffic
            self.adaptive_timings["ns_green"] = max(self.adaptive_timings["ns_green"],
                                                    int(self.default_timings["ns_green"] * 1.2))
            self.adaptive_timings["ew_green"] = max(self.adaptive_timings["ew_green"],
                                                    int(self.default_timings["ew_green"] * 1.2))
            logger.info("Increased green times due to rain")

        if sensor_data.get("light_level", 100) < 30:
            # It's dark - ensure yellow phases are longer for safety
            self.adaptive_timings["ns_yellow"] = 6
            self.adaptive_timings["ew_yellow"] = 6
            logger.info("Increased yellow times due to low light conditions")
        else:
            # Reset to default
            self.adaptive_timings["ns_yellow"] = self.default_timings["ns_yellow"]
            self.adaptive_timings["ew_yellow"] = self.default_timings["ew_yellow"]


class SensorData:
    """Simulates and processes data from various sensors"""

    def __init__(self):
        self.temperature = 25  # in Celsius
        self.is_raining = False
        self.light_level = 100  # 0-100, where 100 is full daylight
        logger.info("Sensor data module initialized")

    def update(self):
        """Simulate updates from sensors"""
        # Simulate temperature changes
        self.temperature += random.uniform(-0.5, 0.5)

        # Simulate rain with 5% chance
        if random.random() < 0.05:
            self.is_raining = not self.is_raining

        # Simulate light level changes (day/night cycle)
        current_hour = datetime.now().hour
        if 6 <= current_hour < 18:  # Daytime
            self.light_level = max(
                70, min(100, self.light_level + random.uniform(-5, 5)))
        else:  # Nighttime
            self.light_level = max(
                5, min(30, self.light_level + random.uniform(-3, 3)))

        logger.debug(
            f"Sensor data updated: temp={self.temperature:.1f}°C, rain={self.is_raining}, light={self.light_level:.1f}%")
        return {
            "temperature": self.temperature,
            "is_raining": self.is_raining,
            "light_level": self.light_level
        }


class VideoRecorder:
    """Handles video recording and saving"""
    
    def __init__(self, output_dir="recordings"):
        """
        Initialize the video recorder
        
        Args:
            output_dir: Directory where videos will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.current_writer = None
        self.current_filename = None
        self.recording = False
        self.frame_width = None
        self.frame_height = None
        self.fps = 10  # Frames per second for recording
        
        logger.info(f"Video recorder initialized. Saving to: {self.output_dir.absolute()}")
    
    def start_recording(self, frame):
        """
        Start a new video recording session
        
        Args:
            frame: First frame to initialize video dimensions
        """
        if self.recording:
            self.stop_recording()
        
        # Get frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = self.output_dir / f"traffic_recording_{timestamp}.avi"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.current_writer = cv2.VideoWriter(
            str(self.current_filename), 
            fourcc, 
            self.fps, 
            (self.frame_width, self.frame_height)
        )
        
        self.recording = True
        logger.info(f"Started recording: {self.current_filename}")
        return str(self.current_filename)
    
    def write_frame(self, frame):
        """
        Write a frame to the current recording
        
        Args:
            frame: Frame to write
        """
        if self.recording and self.current_writer is not None:
            # Resize if needed
            if frame.shape[:2] != (self.frame_height, self.frame_width):
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            self.current_writer.write(frame)
    
    def stop_recording(self):
        """Stop the current recording session"""
        if self.current_writer is not None:
            self.current_writer.release()
            self.current_writer = None
            
            if self.current_filename and self.current_filename.exists():
                file_size = self.current_filename.stat().st_size / (1024 * 1024)  # Size in MB
                logger.info(f"Stopped recording: {self.current_filename} ({file_size:.2f} MB)")
            
            self.current_filename = None
        
        self.recording = False
    
    def get_recording_info(self):
        """
        Get information about the current recording
        
        Returns:
            dict: Recording information
        """
        if self.recording:
            return {
                "recording": True,
                "filename": str(self.current_filename) if self.current_filename else None,
                "frame_size": (self.frame_width, self.frame_height),
                "fps": self.fps
            }
        else:
            return {"recording": False}
    
    def list_recordings(self):
        """
        List all saved recordings
        
        Returns:
            list: List of recording file paths
        """
        recordings = list(self.output_dir.glob("traffic_recording_*.avi"))
        recordings.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return recordings


class SmartTrafficSystem:
    """Main class for the smart traffic system"""

    def __init__(self, camera_source=0, auto_record=True, record_interval=60):
        """
        Initialize the smart traffic system

        Args:
            camera_source: Source for the camera feed (0 for webcam, or file path)
            auto_record: Automatically record video
            record_interval: Recording interval in seconds (0 for continuous recording)
        """
        self.camera_source = camera_source
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        # Auto recording settings
        self.auto_record = auto_record
        self.record_interval = record_interval
        self.last_recording_start = None

        # Initialize detectors
        self.pedestrian_detector = PedestrianDetector()
        self.vehicle_detector = VehicleDetector()

        # Initialize intersection controller
        self.intersection = IntersectionController()

        # Initialize sensor data
        self.sensor_data = SensorData()
        
        # Initialize video recorder
        self.video_recorder = VideoRecorder()

        # Traffic data for each direction
        self.traffic_data = {
            "north": {"vehicles": 0, "pedestrians": 0},
            "south": {"vehicles": 0, "pedestrians": 0},
            "east": {"vehicles": 0, "pedestrians": 0},
            "west": {"vehicles": 0, "pedestrians": 0}
        }

        logger.info("Smart Traffic System initialized")

    def start(self):
        """Start the traffic system"""
        if self.is_running:
            logger.warning("System is already running")
            return

        self.is_running = True

        # Start the camera thread
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        # Start the processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Start the traffic light simulation thread
        self.traffic_thread = threading.Thread(target=self._traffic_light_loop)
        self.traffic_thread.daemon = True
        self.traffic_thread.start()

        logger.info("Smart Traffic System started")

    def stop(self):
        """Stop the traffic system"""
        self.is_running = False
        
        # Stop video recording
        if self.video_recorder.recording:
            self.video_recorder.stop_recording()
            
        logger.info("Smart Traffic System stopping...")

    def _camera_loop(self):
        """Camera capture loop that runs in a separate thread"""
        try:
            # Use the webcam or a video file as the camera source
            cap = cv2.VideoCapture(self.camera_source)

            if not cap.isOpened():
                logger.error(
                    f"Could not open camera source: {self.camera_source}")
                # Use simulated data for testing
                self._simulate_camera_loop()
                return

            logger.info(f"Camera connected to source: {self.camera_source}")

            while self.is_running:
                ret, frame = cap.read()

                if not ret:
                    logger.warning("Failed to capture frame")
                    # If it's a video file that ended, restart it
                    if isinstance(self.camera_source, str):
                        cap = cv2.VideoCapture(self.camera_source)
                        continue
                    break

                # Put the frame in the queue for processing
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

                # Add a small delay to simulate realistic frame rate
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in camera loop: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()

    def _simulate_camera_loop(self):
        """Simulate camera data when no camera is available"""
        logger.info("Using simulated camera data")

        # Create a blank frame (640x480)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        while self.is_running:
            # Create a copy of the blank frame
            frame = blank_frame.copy()

            # Draw some random shapes to simulate objects
            for _ in range(random.randint(1, 5)):
                # Random rectangle representing a vehicle
                x, y = random.randint(0, 600), random.randint(0, 440)
                cv2.rectangle(frame, (x, y), (x + 40, y + 40), (0, 0, 255), -1)

            for _ in range(random.randint(0, 3)):
                # Random circle representing a pedestrian
                x, y = random.randint(0, 620), random.randint(0, 460)
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Put the frame in the queue for processing
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            # Add a small delay
            time.sleep(0.5)

    def _handle_recording(self, processed_frame):
        """
        Handle video recording logic
        
        Args:
            processed_frame: The processed frame to record
        """
        if not self.auto_record:
            return
        
        # Start recording if not already recording
        if not self.video_recorder.recording:
            self.video_recorder.start_recording(processed_frame)
            self.last_recording_start = time.time()
        
        # Check if we need to stop and start a new recording based on interval
        elif self.record_interval > 0 and self.last_recording_start is not None:
            if time.time() - self.last_recording_start >= self.record_interval:
                # Stop current recording and start a new one
                self.video_recorder.stop_recording()
                self.video_recorder.start_recording(processed_frame)
                self.last_recording_start = time.time()
                logger.info(f"Started new recording segment")
        
        # Write the frame
        self.video_recorder.write_frame(processed_frame)

    def _processing_loop(self):
        """Process frames in a separate thread"""
        while self.is_running:
            try:
                # Get the next frame from the queue
                if self.frame_queue.empty():
                    time.sleep(0.1)
                    continue

                frame = self.frame_queue.get()

                # Process the frame
                pedestrian_boxes, pedestrian_count = self.pedestrian_detector.detect(
                    frame)
                vehicle_boxes, vehicle_count = self.vehicle_detector.detect(
                    frame)

                # Update sensor data
                sensor_data = self.sensor_data.update()

                # For demo purposes, we'll simulate traffic in different directions
                # Reset traffic data
                for direction in self.traffic_data:
                    self.traffic_data[direction]["vehicles"] = 0
                    self.traffic_data[direction]["pedestrians"] = 0

                # Distribute traffic randomly (for simulation)
                directions = ["north", "south", "east", "west"]

                # Distribute vehicles
                for _ in range(vehicle_count):
                    direction = random.choice(directions)
                    self.traffic_data[direction]["vehicles"] += 1

                # Distribute pedestrians
                for _ in range(pedestrian_count):
                    direction = random.choice(directions)
                    self.traffic_data[direction]["pedestrians"] += 1

                # Store the results
                result = {
                    "frame": frame,
                    "traffic_data": self.traffic_data,
                    "sensor_data": sensor_data,
                    "timestamp": time.time()
                }

                # Put the result in the queue
                if not self.result_queue.full():
                    self.result_queue.put(result)

                logger.info(
                    f"Processed frame: {pedestrian_count} pedestrians, {vehicle_count} vehicles")

                # Draw boxes around detected objects for visualization
                processed_frame = frame.copy()

                # Draw pedestrian boxes
                for (x, y, w, h) in pedestrian_boxes:
                    cv2.rectangle(processed_frame, (x, y),
                                  (x + w, y + h), (0, 255, 0), 2)

                # Draw vehicle boxes
                for (x, y, w, h) in vehicle_boxes:
                    cv2.rectangle(processed_frame, (x, y),
                                  (x + w, y + h), (255, 0, 0), 2)

                # Display information
                cv2.putText(processed_frame, f"Pedestrians: {pedestrian_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Vehicles: {vehicle_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Add recording indicator
                if self.video_recorder.recording and self.auto_record:
                    cv2.putText(processed_frame, "RECORDING", (processed_frame.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Draw red recording dot
                    cv2.circle(processed_frame, (processed_frame.shape[1] - 170, 25), 5, (0, 0, 255), -1)

                # Show traffic directions info
                y_offset = 90
                for direction in directions:
                    cv2.putText(processed_frame,
                                f"{direction.capitalize()}: {self.traffic_data[direction]['vehicles']} v, {self.traffic_data[direction]['pedestrians']} p",
                                (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25

                # Handle video recording
                self._handle_recording(processed_frame)

                # Show the processed frame
                cv2.imshow("Smart Traffic System", processed_frame)

                # Display traffic light status
                self._display_traffic_lights()

                # Break the loop on 'q' key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop()
                elif key == ord('r'):  # Manual recording toggle
                    if self.video_recorder.recording:
                        self.video_recorder.stop_recording()
                        logger.info("Manual recording stopped")
                    else:
                        self.video_recorder.start_recording(processed_frame)
                        logger.info("Manual recording started")
                elif key == ord('s'):  # Screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = self.video_recorder.output_dir / f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(str(screenshot_path), processed_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")

            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")

    def _traffic_light_loop(self):
        """Simulate and update traffic lights"""
        while self.is_running:
            try:
                # Get the latest detection results
                if not self.result_queue.empty():
                    result = self.result_queue.get()
                    traffic_data = result["traffic_data"]
                    sensor_data = result["sensor_data"]

                    # Check for emergency vehicles (simulated randomly)
                    if random.random() < 0.01:  # 1% chance of emergency
                        emergency_direction = random.choice(
                            ["north", "south", "east", "west"])
                        logger.info(
                            f"Emergency vehicle detected approaching from {emergency_direction}")
                        self.intersection.adjust_for_emergency(
                            emergency_direction)
                    else:
                        # Normal traffic adjustment
                        self.intersection.adjust_timing(traffic_data)
                        self.intersection.apply_environmental_adjustments(
                            sensor_data)

                # Update intersection controller
                self.intersection.update()

                time.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Error in traffic light loop: {str(e)}")

    def _display_traffic_lights(self):
        """Display traffic light status"""
        # Create a blank image for the traffic lights
        traffic_display = np.zeros((300, 400, 3), dtype=np.uint8)

        # Draw intersection layout
        cv2.rectangle(traffic_display, (150, 100), (250, 200),
                      (50, 50, 50), -1)  # Intersection
        cv2.rectangle(traffic_display, (0, 125), (150, 175),
                      (40, 40, 40), -1)    # West road
        cv2.rectangle(traffic_display, (250, 125), (400, 175),
                      (40, 40, 40), -1)  # East road
        cv2.rectangle(traffic_display, (175, 0), (225, 100),
                      (40, 40, 40), -1)    # North road
        cv2.rectangle(traffic_display, (175, 200), (225, 300),
                      (40, 40, 40), -1)  # South road

        # Display each traffic light
        positions = {
            "north": (200, 75),
            "east": (275, 150),
            "south": (200, 225),
            "west": (125, 150)
        }

        # Current phase info
        phase_names = {
            0: "N/S Green, E/W Red",
            1: "N/S Yellow, E/W Red",
            2: "N/S Red, E/W Green",
            3: "N/S Red, E/W Yellow"
        }

        # Draw phase info
        cv2.putText(traffic_display, f"Phase: {self.intersection.current_phase} - {phase_names[self.intersection.current_phase]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(traffic_display, f"Time Remaining: {self.intersection.phase_remaining_time}s",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw traffic lights
        for direction, light in self.intersection.traffic_lights.items():
            x, y = positions[direction]

            # Choose color based on state
            if light.current_state == "green":
                color = (0, 255, 0)
            elif light.current_state == "yellow":
                color = (0, 255, 255)
            else:  # red
                color = (0, 0, 255)

            # Draw a circle representing the traffic light
            cv2.circle(traffic_display, (x, y), 15, color, -1)

            # Display direction
            cv2.putText(traffic_display, direction,
                        (x - 20, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Show the traffic light display
        cv2.imshow("Traffic Lights", traffic_display)


def main():
    """Main function to run the smart traffic system"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Smart Traffic Management System')
    parser.add_argument('--camera', type=int, default=0, help='Camera source (default: 0)')
    parser.add_argument('--no-record', action='store_true', help='Disable auto recording')
    parser.add_argument('--record-interval', type=int, default=60, help='Recording interval in seconds (0 for continuous)')
    parser.add_argument('--list-recordings', action='store_true', help='List existing recordings')
    
    args = parser.parse_args()
    
    # List recordings if requested
    if args.list_recordings:
        recorder = VideoRecorder()
        recordings = recorder.list_recordings()
        if recordings:
            print("\nExisting Recordings:")
            print("-" * 50)
            for i, recording in enumerate(recordings, 1):
                size_mb = recording.stat().st_size / (1024 * 1024)
                modified_time = datetime.fromtimestamp(recording.stat().st_mtime)
                print(f"{i}. {recording.name}")
                print(f"   Size: {size_mb:.2f} MB")
                print(f"   Date: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        else:
            print("No recordings found.")
        return
    
    # Create and start the smart traffic system
    system = SmartTrafficSystem(
        camera_source=args.camera,
        auto_record=not args.no_record,
        record_interval=args.record_interval
    )
    system.start()
    
    print("\n" + "="*50)
    print("Smart Traffic System Started")
    print("-" * 50)
    print("Controls:")
    print("  Q - Quit the application")
    print("  R - Start/Stop manual recording")
    print("  S - Take screenshot")
    print("-" * 50)
    if not args.no_record:
        print(f"Auto-recording: Enabled (Interval: {args.record_interval}s)")
    else:
        print("Auto-recording: Disabled")
    print(f"Recordings saved to: recordings/")
    print("="*50 + "\n")
    
    try:
        # Keep the main thread alive
        while system.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle Ctrl+C
        print("\nShutting down...")
        system.stop()
    finally:
        # Clean up
        cv2.destroyAllWindows()
        logger.info("Smart Traffic System stopped")


if __name__ == "__main__":
    main()