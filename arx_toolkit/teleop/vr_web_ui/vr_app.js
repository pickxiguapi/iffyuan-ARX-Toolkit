/**
 * ARX VR Teleop — A-Frame controller data capture.
 *
 * Adapted from XLeVR vr_app.js.
 * Captures Quest 3 controller position/quaternion/trigger/grip per frame
 * and sends JSON over WSS to the Python VRTeleop server.
 *
 * URL query params:
 *   ?swap=1  — swap trigger/grip roles (trigger=arm, grip=gripper)
 */

// --- Configuration from URL query params ---
const URL_PARAMS = new URLSearchParams(window.location.search);
const SWAP_BUTTONS = URL_PARAMS.get('swap') === '1';

// --- Speed levels (5 levels) ---
const SPEED_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0];

AFRAME.registerComponent('controller-updater', {
  init: function () {
    console.log("ARX VR controller-updater initialized.");
    console.log(`Button mode: ${SWAP_BUTTONS ? 'trigger=arm, grip=gripper' : 'grip=arm, trigger=gripper'}`);

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

    // --- Speed level (0=slowest, 4=fastest) ---
    this.speedLevel = 0;  // default: slowest (0.2x)

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

    // --- Helper: bind arm activation events ---
    // In default mode: grip = arm activate, trigger = gripper
    // In swap mode:    trigger = arm activate, grip = gripper
    const armDownEvent = SWAP_BUTTONS ? 'triggerdown' : 'gripdown';
    const armUpEvent   = SWAP_BUTTONS ? 'triggerup'   : 'gripup';

    // Left hand — arm activation
    this.leftHand.addEventListener(armDownEvent, () => {
      this.leftGripDown = true;
      if (this.leftHand.object3D.visible) {
        this.leftGripInitialQuaternion = this.leftHand.object3D.quaternion.clone();
      }
    });
    this.leftHand.addEventListener(armUpEvent, () => {
      this.leftGripDown = false;
      this.leftGripInitialQuaternion = null;
      this.leftZAxisRotation = 0;
    });

    // Right hand — arm activation
    this.rightHand.addEventListener(armDownEvent, () => {
      this.rightGripDown = true;
      if (this.rightHand.object3D.visible) {
        this.rightGripInitialQuaternion = this.rightHand.object3D.quaternion.clone();
      }
    });
    this.rightHand.addEventListener(armUpEvent, () => {
      this.rightGripDown = false;
      this.rightGripInitialQuaternion = null;
      this.rightZAxisRotation = 0;
    });

    // Left/right hand — gripper (the OTHER button)
    const gripperDownEvent = SWAP_BUTTONS ? 'gripdown' : 'triggerdown';
    const gripperUpEvent   = SWAP_BUTTONS ? 'gripup'   : 'triggerup';

    this.leftHand.addEventListener(gripperDownEvent, () => { this.leftTriggerDown = true; });
    this.leftHand.addEventListener(gripperUpEvent,   () => { this.leftTriggerDown = false; });
    this.rightHand.addEventListener(gripperDownEvent, () => { this.rightTriggerDown = true; });
    this.rightHand.addEventListener(gripperUpEvent,   () => { this.rightTriggerDown = false; });

    // --- Speed control: X = speed up, Y = speed down (left controller) ---
    this.leftHand.addEventListener('xbuttondown', () => {
      this.speedLevel = Math.min(this.speedLevel + 1, 4);
      console.log(`Speed UP: level ${this.speedLevel + 1}/5 (scale=${SPEED_LEVELS[this.speedLevel]})`);
    });
    this.leftHand.addEventListener('ybuttondown', () => {
      this.speedLevel = Math.max(this.speedLevel - 1, 0);
      console.log(`Speed DOWN: level ${this.speedLevel + 1}/5 (scale=${SPEED_LEVELS[this.speedLevel]})`);
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
          headset: headset,
          speedLevel: this.speedLevel
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

  // "Enter VR" button — full-screen, unmissable
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
    btn.textContent = '进入 VR 控制';
    btn.style.cssText = `
      position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
      font-size: 48px; font-weight: bold;
      background: #2ecc40; color: white; border: none;
      cursor: pointer; z-index: 99999;
      display: flex; align-items: center; justify-content: center;
      letter-spacing: 4px;
      animation: pulse 2s ease-in-out infinite;
    `;

    // Add pulse animation
    const style = document.createElement('style');
    style.textContent = `
      @keyframes pulse {
        0%, 100% { background: #2ecc40; }
        50%      { background: #27ae60; }
      }
    `;
    document.head.appendChild(style);

    btn.onclick = () => {
      const sceneEl = document.querySelector('a-scene');
      if (sceneEl) {
        sceneEl.enterVR(true).then(() => {
          // Beep confirmation sound
          try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.frequency.value = 880;
            gain.gain.value = 0.3;
            osc.start();
            osc.stop(ctx.currentTime + 0.15);
          } catch(e) { /* audio not critical */ }
        }).catch(err => {
          console.error('Failed to enter VR:', err);
          alert('Failed to start AR session: ' + err.message);
        });
      }
    };
    document.body.appendChild(btn);

    const sceneEl = document.querySelector('a-scene');
    if (sceneEl) {
      sceneEl.addEventListener('enter-vr', () => { btn.style.display = 'none'; });
      sceneEl.addEventListener('exit-vr', () => { btn.style.display = 'flex'; });
    }
  }).catch(err => console.error('XR check error:', err));
}
