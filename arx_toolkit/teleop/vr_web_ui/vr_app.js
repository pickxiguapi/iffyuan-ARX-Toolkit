/**
 * ARX VR Teleop — A-Frame controller data capture.
 *
 * Adapted from XLeVR vr_app.js.
 * Captures Quest 3 controller position/quaternion/trigger/grip per frame
 * and sends JSON over WSS to the Python VRTeleop server.
 */

AFRAME.registerComponent('controller-updater', {
  init: function () {
    console.log("ARX VR controller-updater initialized.");

    this.leftHand = document.querySelector('#leftHand');
    this.rightHand = document.querySelector('#rightHand');
    this.leftHandInfoText = document.querySelector('#leftHandInfo');
    this.rightHandInfoText = document.querySelector('#rightHandInfo');
    this.headset = document.querySelector('#headset');
    this.headsetInfoText = document.querySelector('#headsetInfo');

    // --- WebSocket ---
    this.websocket = null;
    this.leftGripDown = false;
    this.rightGripDown = false;
    this.leftTriggerDown = false;
    this.rightTriggerDown = false;

    // --- Quaternion tracking for display ---
    this.leftGripInitialQuaternion = null;
    this.rightGripInitialQuaternion = null;
    this.leftZAxisRotation = 0;
    this.rightZAxisRotation = 0;

    // --- WebSocket connection ---
    const serverHostname = window.location.hostname;
    const websocketPort = 8442;
    const websocketUrl = `wss://${serverHostname}:${websocketPort}`;
    console.log(`Connecting WebSocket: ${websocketUrl}`);

    try {
      this.websocket = new WebSocket(websocketUrl);
      this.websocket.onopen = () => {
        console.log(`WebSocket connected to ${websocketUrl}`);
        this.reportVRStatus(true);
      };
      this.websocket.onerror = (event) => {
        console.error('WebSocket Error:', event);
        this.reportVRStatus(false);
      };
      this.websocket.onclose = (event) => {
        console.log(`WebSocket closed. Clean: ${event.wasClean}`);
        this.websocket = null;
        this.reportVRStatus(false);
      };
    } catch (error) {
      console.error('WebSocket creation failed:', error);
      this.reportVRStatus(false);
    }

    this.reportVRStatus = (connected) => {
      if (typeof updateVRStatus === 'function') {
        updateVRStatus(connected);
      }
    };

    if (!this.leftHand || !this.rightHand) {
      console.error("Controller entities not found!");
      return;
    }

    // Rotate info text for readability
    const textRotation = '-90 0 0';
    if (this.leftHandInfoText) this.leftHandInfoText.setAttribute('rotation', textRotation);
    if (this.rightHandInfoText) this.rightHandInfoText.setAttribute('rotation', textRotation);

    // Axis indicators
    this.createAxisIndicators();

    // --- Quaternion Z-axis rotation helper ---
    this.calculateZAxisRotation = (currentQuaternion, initialQuaternion) => {
      const relativeQuat = new THREE.Quaternion();
      relativeQuat.multiplyQuaternions(currentQuaternion, initialQuaternion.clone().invert());
      const forwardDirection = new THREE.Vector3(0, 0, 1);
      forwardDirection.applyQuaternion(currentQuaternion);
      const angle = 2 * Math.acos(Math.abs(relativeQuat.w));
      if (angle < 0.0001) return 0;
      const sinHalfAngle = Math.sqrt(1 - relativeQuat.w * relativeQuat.w);
      const rotationAxis = new THREE.Vector3(
        relativeQuat.x / sinHalfAngle,
        relativeQuat.y / sinHalfAngle,
        relativeQuat.z / sinHalfAngle
      );
      const projectedComponent = rotationAxis.dot(forwardDirection);
      let degrees = THREE.MathUtils.radToDeg(angle * projectedComponent);
      while (degrees > 180) degrees -= 360;
      while (degrees < -180) degrees += 360;
      return degrees;
    };

    // --- Event listeners ---
    this.leftHand.addEventListener('triggerdown', () => { this.leftTriggerDown = true; });
    this.leftHand.addEventListener('triggerup', () => { this.leftTriggerDown = false; });
    this.leftHand.addEventListener('gripdown', () => {
      this.leftGripDown = true;
      if (this.leftHand.object3D.visible) {
        this.leftGripInitialQuaternion = this.leftHand.object3D.quaternion.clone();
      }
    });
    this.leftHand.addEventListener('gripup', () => {
      this.leftGripDown = false;
      this.leftGripInitialQuaternion = null;
      this.leftZAxisRotation = 0;
    });

    this.rightHand.addEventListener('triggerdown', () => { this.rightTriggerDown = true; });
    this.rightHand.addEventListener('triggerup', () => { this.rightTriggerDown = false; });
    this.rightHand.addEventListener('gripdown', () => {
      this.rightGripDown = true;
      if (this.rightHand.object3D.visible) {
        this.rightGripInitialQuaternion = this.rightHand.object3D.quaternion.clone();
      }
    });
    this.rightHand.addEventListener('gripup', () => {
      this.rightGripDown = false;
      this.rightGripInitialQuaternion = null;
      this.rightZAxisRotation = 0;
    });
  },

  createAxisIndicators: function() {
    const colors = [['#ff0000','X'], ['#00ff00','Y'], ['#0000ff','Z']];
    const positions = [
      ['0.04 0 0', '0 0 90'],
      ['0 0.04 0', '0 0 0'],
      ['0 0 0.04', '90 0 0']
    ];
    const tipPositions = ['0.055 0 0', '0 0.055 0', '0 0 0.055'];

    [this.leftHand, this.rightHand].forEach(hand => {
      for (let i = 0; i < 3; i++) {
        const axis = document.createElement('a-cylinder');
        axis.setAttribute('height', '0.08');
        axis.setAttribute('radius', '0.003');
        axis.setAttribute('color', colors[i][0]);
        axis.setAttribute('position', positions[i][0]);
        axis.setAttribute('rotation', positions[i][1]);
        hand.appendChild(axis);

        const tip = document.createElement('a-cone');
        tip.setAttribute('height', '0.015');
        tip.setAttribute('radius-bottom', '0.008');
        tip.setAttribute('radius-top', '0');
        tip.setAttribute('color', colors[i][0]);
        tip.setAttribute('position', tipPositions[i]);
        tip.setAttribute('rotation', positions[i][1]);
        hand.appendChild(tip);
      }
    });
  },

  tick: function () {
    if (!this.leftHand || !this.rightHand) return;

    const leftController = { hand: 'left', position: null, rotation: null, quaternion: null, gripActive: false, trigger: 0 };
    const rightController = { hand: 'right', position: null, rotation: null, quaternion: null, gripActive: false, trigger: 0 };
    const headset = { position: null, rotation: null, quaternion: null };

    // --- Left hand ---
    if (this.leftHand && this.leftHand.object3D) {
      const pos = this.leftHand.object3D.position;
      const rotE = this.leftHand.object3D.rotation;
      const rotDeg = {
        x: THREE.MathUtils.radToDeg(rotE.x),
        y: THREE.MathUtils.radToDeg(rotE.y),
        z: THREE.MathUtils.radToDeg(rotE.z)
      };

      if (this.leftGripDown && this.leftGripInitialQuaternion) {
        this.leftZAxisRotation = this.calculateZAxisRotation(
          this.leftHand.object3D.quaternion, this.leftGripInitialQuaternion
        );
      }

      let txt = `L Pos: ${pos.x.toFixed(2)} ${pos.y.toFixed(2)} ${pos.z.toFixed(2)}`;
      if (this.leftGripDown) txt += `\\nZ-Rot: ${this.leftZAxisRotation.toFixed(1)}°`;
      if (this.leftHandInfoText) this.leftHandInfoText.setAttribute('value', txt);

      leftController.position = { x: pos.x, y: pos.y, z: pos.z };
      leftController.rotation = rotDeg;
      leftController.quaternion = {
        x: this.leftHand.object3D.quaternion.x,
        y: this.leftHand.object3D.quaternion.y,
        z: this.leftHand.object3D.quaternion.z,
        w: this.leftHand.object3D.quaternion.w
      };
      leftController.trigger = this.leftTriggerDown ? 1 : 0;
      leftController.gripActive = this.leftGripDown;
    }

    // --- Right hand ---
    if (this.rightHand && this.rightHand.object3D) {
      const pos = this.rightHand.object3D.position;
      const rotE = this.rightHand.object3D.rotation;
      const rotDeg = {
        x: THREE.MathUtils.radToDeg(rotE.x),
        y: THREE.MathUtils.radToDeg(rotE.y),
        z: THREE.MathUtils.radToDeg(rotE.z)
      };

      if (this.rightGripDown && this.rightGripInitialQuaternion) {
        this.rightZAxisRotation = this.calculateZAxisRotation(
          this.rightHand.object3D.quaternion, this.rightGripInitialQuaternion
        );
      }

      let txt = `R Pos: ${pos.x.toFixed(2)} ${pos.y.toFixed(2)} ${pos.z.toFixed(2)}`;
      if (this.rightGripDown) txt += `\\nZ-Rot: ${this.rightZAxisRotation.toFixed(1)}°`;
      if (this.rightHandInfoText) this.rightHandInfoText.setAttribute('value', txt);

      rightController.position = { x: pos.x, y: pos.y, z: pos.z };
      rightController.rotation = rotDeg;
      rightController.quaternion = {
        x: this.rightHand.object3D.quaternion.x,
        y: this.rightHand.object3D.quaternion.y,
        z: this.rightHand.object3D.quaternion.z,
        w: this.rightHand.object3D.quaternion.w
      };
      rightController.trigger = this.rightTriggerDown ? 1 : 0;
      rightController.gripActive = this.rightGripDown;
    }

    // --- Headset ---
    if (this.headset && this.headset.object3D) {
      const hp = this.headset.object3D.position;
      const hr = this.headset.object3D.rotation;
      headset.position = { x: hp.x, y: hp.y, z: hp.z };
      headset.rotation = {
        x: THREE.MathUtils.radToDeg(hr.x),
        y: THREE.MathUtils.radToDeg(hr.y),
        z: THREE.MathUtils.radToDeg(hr.z)
      };
      headset.quaternion = {
        x: this.headset.object3D.quaternion.x,
        y: this.headset.object3D.quaternion.y,
        z: this.headset.object3D.quaternion.z,
        w: this.headset.object3D.quaternion.w
      };
      if (this.headsetInfoText) {
        this.headsetInfoText.setAttribute('value',
          `Head: ${hp.x.toFixed(2)} ${hp.y.toFixed(2)} ${hp.z.toFixed(2)}`);
      }
    }

    // --- Send over WebSocket ---
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      const hasLeft = leftController.position !== null;
      const hasRight = rightController.position !== null;
      if (hasLeft || hasRight) {
        this.websocket.send(JSON.stringify({
          timestamp: Date.now(),
          leftController: leftController,
          rightController: rightController,
          headset: headset
        }));
      }
    }
  }
});


// Scene setup
document.addEventListener('DOMContentLoaded', () => {
  const scene = document.querySelector('a-scene');
  if (scene) {
    scene.addEventListener('controllerconnected', (evt) => {
      console.log('Controller connected:', evt.detail.name, evt.detail.component.data.hand);
    });
    scene.addEventListener('controllerdisconnected', (evt) => {
      console.log('Controller disconnected:', evt.detail.name);
    });
    if (scene.hasLoaded) {
      scene.setAttribute('controller-updater', '');
    } else {
      scene.addEventListener('loaded', () => {
        scene.setAttribute('controller-updater', '');
      });
    }
  }

  // "Start Controller Tracking" button for Quest 3
  addControllerTrackingButton();
});

function addControllerTrackingButton() {
  if (!navigator.xr) {
    console.warn('WebXR not supported.');
    return;
  }
  navigator.xr.isSessionSupported('immersive-ar').then((supported) => {
    if (!supported) {
      console.warn('immersive-ar not supported.');
      return;
    }

    const btn = document.createElement('button');
    btn.id = 'start-tracking-button';
    btn.textContent = 'Start Controller Tracking';
    btn.style.cssText = `
      position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
      padding:20px 40px; font-size:20px; font-weight:bold;
      background:#4CAF50; color:white; border:none; border-radius:8px;
      cursor:pointer; z-index:9999; box-shadow:0 4px 8px rgba(0,0,0,0.3);
    `;
    btn.onclick = () => {
      const sceneEl = document.querySelector('a-scene');
      if (sceneEl) {
        sceneEl.enterVR(true).catch(err => {
          console.error('Failed to enter VR:', err);
          alert('Failed to start AR session: ' + err.message);
        });
      }
    };
    document.body.appendChild(btn);

    const sceneEl = document.querySelector('a-scene');
    if (sceneEl) {
      sceneEl.addEventListener('enter-vr', () => { btn.style.display = 'none'; });
      sceneEl.addEventListener('exit-vr', () => { btn.style.display = 'block'; });
    }
  }).catch(err => console.error('XR check error:', err));
}
