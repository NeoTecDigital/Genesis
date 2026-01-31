<script>
  import { onMount, onDestroy } from 'svelte';
  import { clustersStore, memoriesStore, activationStore } from '../stores/websocket.js';
  import { filterState } from '../stores/filters.js';
  import * as THREE from 'three';
  
  let canvasContainer;
  let scene, camera, renderer;
  let particleSystem, wavetableLines;
  let controls;
  
  // Filter state
  let selectedFilters = {
    gamma: true,
    iota: true,
    tau: true,
    epsilon: true,
    protoIdentity: true,
    instances: true
  };
  
  let filterUnsubscribe;
  
  $: clusters = $clustersStore;
  $: memories = $memoriesStore;
  $: activation = $activationStore;
  
  function updateParticles() {
    if (!scene) return;
    
    // Remove old particle system
    if (particleSystem) {
      scene.remove(particleSystem);
      particleSystem.geometry.dispose();
      particleSystem.material.dispose();
    }
    
    // Use memories (point cloud) if available, otherwise fall back to clusters
    const dataMemories = memories && memories.length > 0 ? memories : null;
    const dataClusters = clusters && clusters.length > 0 ? clusters : null;
    
    let positions = [];
    let colors = [];
    let sizes = [];
    
    if (dataMemories && dataMemories.length > 0) {
      // Render full point cloud of memories - 1 particle per memory
      const multiplier = 1; // 1 particle per memory
      const totalParticles = dataMemories.length * multiplier;
      console.log(`🔄 Updating point cloud: ${dataMemories.length} memories → ${totalParticles} particles`);
      
      // Scale factor for entire particle system
      const positionScale = 0.8; // Scale to 0.8
      
      for (let i = 0; i < dataMemories.length; i++) {
        const memory = dataMemories[i];
        const basePos = memory.position || [0, 0, 0];
        const coherence = memory.coherence || 0.7;
        
        // Generate 1 particle per memory (10x less dense)
        for (let j = 0; j < multiplier; j++) {
          // Reduced offset radius for 10x less density
          const offset = 0.15; // Reduced from 1.5 to 0.15 (10x smaller cluster spread)
          const offsetX = (Math.random() - 0.5) * offset;
          const offsetY = (Math.random() - 0.5) * offset;
          const offsetZ = (Math.random() - 0.5) * offset;
          
        // Arrange particles in brain-like shape (spherical/bilaterally symmetrical)
        // Transform flat distribution into brain-like volume
        const baseX = (basePos[0] || 0) * positionScale;
        const baseY = (basePos[1] || 0) * positionScale;
        const baseZ = (basePos[2] || 0) * positionScale;
        
        // Create brain-like shape: two hemispheres with central division
        // Map to spherical coordinates for brain shape
        const radius = Math.sqrt(baseX * baseX + baseY * baseY + baseZ * baseZ);
        const maxRadius = 8.0; // Brain size
        const normalizedRadius = Math.min(radius / maxRadius, 1.0);
        
        // Create bilateral symmetry (two hemispheres)
        const hemisphere = baseX > 0 ? 1 : -1; // Left/right hemisphere
        const brainX = hemisphere * Math.abs(baseX) * (0.8 + normalizedRadius * 0.2);
        const brainY = baseY * (0.9 + normalizedRadius * 0.1); // Slight vertical compression
        const brainZ = baseZ * (0.85 + normalizedRadius * 0.15); // Depth
        
        // Add offset for cluster density
        positions.push(
          brainX + offsetX,
          brainY + offsetY,
          brainZ + offsetZ
        );
          
        // Color: Cyan/Electric Blue particles (brain theme)
        colors.push(0.0, 0.3 + coherence * 0.7, 0.6 + coherence * 0.4);
          sizes.push(0.1 + coherence * 0.05); // Point size
        }
      }
    } else if (dataClusters && dataClusters.length > 0) {
      // Fallback: render cluster centroids
      console.log(`🔄 Updating cluster centroids: ${dataClusters.length} clusters`);
      
      // Scale factor for entire particle system
      const positionScale = 0.8; // Scale to 0.8
      
      for (let i = 0; i < dataClusters.length; i++) {
        const cluster = dataClusters[i];
        const pos = cluster.centroid || cluster.position || [0, 0, 0];
        
        // Transform to brain-like shape
        const baseX = (pos[0] || 0) * positionScale;
        const baseY = (pos[1] || 0) * positionScale;
        const baseZ = (pos[2] || 0) * positionScale;
        
        const radius = Math.sqrt(baseX * baseX + baseY * baseY + baseZ * baseZ);
        const maxRadius = 8.0;
        const normalizedRadius = Math.min(radius / maxRadius, 1.0);
        
        const hemisphere = baseX > 0 ? 1 : -1;
        const brainX = hemisphere * Math.abs(baseX) * (0.8 + normalizedRadius * 0.2);
        const brainY = baseY * (0.9 + normalizedRadius * 0.1);
        const brainZ = baseZ * (0.85 + normalizedRadius * 0.15);
        
        positions.push(brainX, brainY, brainZ);
        
        // Color: Cyan/Electric Blue particles
        const coherence = cluster.coherence || 0.7;
        colors.push(0.0, 0.3 + coherence * 0.7, 0.6 + coherence * 0.4);
        sizes.push(0.15 + Math.log((cluster.memory_count || 1) + 1) * 0.02);
      }
    } else {
      // Generate test particles in brain-like shape (10x less dense)
      console.log('🔄 Generating test particles: 50');
      
      for (let i = 0; i < 50; i++) {
        const angle1 = (i / 500) * Math.PI * 2;
        const angle2 = (i / 500) * Math.PI;
        const radius = 5.0;
        
        // Spherical coordinates
        const x = Math.cos(angle1) * Math.sin(angle2) * radius;
        const y = Math.sin(angle1) * Math.sin(angle2) * radius;
        const z = Math.cos(angle2) * radius;
        
        // Transform to brain-like shape (bilateral symmetry)
        const hemisphere = x > 0 ? 1 : -1;
        const normalizedRadius = Math.min(radius / 8.0, 1.0);
        const brainX = hemisphere * Math.abs(x) * (0.8 + normalizedRadius * 0.2);
        const brainY = y * (0.9 + normalizedRadius * 0.1);
        const brainZ = z * (0.85 + normalizedRadius * 0.15);
        
        positions.push(brainX, brainY, brainZ);
        
        colors.push(0.0, 0.5, 1.0); // Cyan/blue for test particles
        sizes.push(0.1);
      }
    }
    
    if (positions.length === 0) return;
    
    // Create geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
    
    // Create shader material for glowing points
    // Note: Three.js already defines 'color' attribute, so we use it directly
    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 }
      },
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        uniform float time;
        
        void main() {
          vColor = color; // Use Three.js built-in color attribute
          vec3 pos = position;
          // Completely static - no animation to eliminate flickering
          vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
          // Static size - no pulsing
          gl_PointSize = size * (300.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        
        void main() {
          vec2 coord = gl_PointCoord - vec2(0.5);
          float dist = length(coord);
          float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
          float glow = exp(-dist * 3.0) * 0.5;
          vec3 finalColor = vColor + vec3(glow);
          gl_FragColor = vec4(finalColor, alpha);
        }
      `,
      transparent: true,
      vertexColors: true
    });
    
    particleSystem = new THREE.Points(geometry, material);
    scene.add(particleSystem);
    
    console.log(`✅ Particles updated: ${positions.length / 3} points`);
  }
  
  // Store wavetable data for animation
  let wavetableData = null;
  
  function updateWavetable() {
    if (!scene || !renderer) return;
    
    // Remove old wavetable
    if (wavetableLines) {
      scene.remove(wavetableLines);
      wavetableLines.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
      if (wavetableLines.material.uniforms?.intensityTexture) {
        wavetableLines.material.uniforms.intensityTexture.value.dispose();
      }
      if (wavetableLines.material.uniforms?.frequencyTexture) {
        wavetableLines.material.uniforms.frequencyTexture.value.dispose();
      }
      wavetableLines = null;
    }
    
    const tableSize = 512; // 512x512 grid
    const filterNames = ['gamma', 'iota', 'tau', 'epsilon', 'protoIdentity', 'instances'];
    
    // Generate wavetable data as textures (much more efficient)
    // Store intensity (w) and frequency in separate textures
    const intensityData = new Float32Array(tableSize * tableSize);
    const frequencyData = new Float32Array(tableSize * tableSize);
    const colorData = new Float32Array(tableSize * tableSize * 3); // RGB per row
    
    // Scale factor for positioning
    const scale = 20.0;
    
    for (let y = 0; y < tableSize; y++) {
      const filterIndex = y % 6;
      const filterName = filterNames[filterIndex];
      const isActive = selectedFilters[filterName];
      const rowColor = getFilterColor(filterIndex);
      
      for (let x = 0; x < tableSize; x++) {
        const idx = y * tableSize + x;
        let intensity = 0;
        let frequency = 1.0;
        
        if (isActive) {
          if (activation?.waveform && activation.waveform.length > 0) {
            // Smooth interpolation in X direction
            const waveformIndex = x % activation.waveform.length;
            const nextXIndex = (x + 1) % activation.waveform.length;
            const tx = (x % activation.waveform.length) / activation.waveform.length;
            const baseWaveX = activation.waveform[waveformIndex] || 0;
            const nextWaveX = activation.waveform[nextXIndex] || 0;
            // Interpolate between samples for smooth X-direction waveform
            const smoothWave = baseWaveX * (1.0 - tx) + nextWaveX * tx;
            
            if (filterIndex === 0) {
              // Gamma: ∅ → 𝟙 (genesis) - Multi-harmonic series with Gaussian envelope
              // Generate brainwave-like pattern: smooth oscillations with harmonics
              const t = (x / tableSize) * Math.PI * 2; // Time/sample position
              let waveform = 0;
              
              // Fundamental frequency (alpha/beta range: 8-30 Hz equivalent)
              const fundamentalFreq = 2.0;
              waveform += Math.sin(t * fundamentalFreq) * 0.4;
              
              // Add harmonics with exponential decay (like gamma_genesis.comp)
              const numHarmonics = 8;
              const harmonicDecay = 0.75;
              for (let h = 2; h <= numHarmonics; h++) {
                const harmonicAmp = Math.pow(harmonicDecay, h - 2) * 0.3;
                waveform += harmonicAmp * Math.sin(t * fundamentalFreq * h);
              }
              
              // Apply Gaussian envelope (frequency domain shape)
              const freqMag = Math.abs((x / tableSize - 0.5) * 2);
              const envelope = Math.exp(-freqMag * freqMag / (2 * 0.4 * 0.4));
              waveform *= envelope;
              
              // Convert to amplitude (0-1 range)
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              
              // Frequency for pulsing: varies by row
              frequency = 1.0 + Math.sin(y * 0.01) * 0.3;
            } else if (filterIndex === 1) {
              // Iota: 𝟙 → n (instantiation) - Frequency-bin modulation
              // Brainwave pattern: modulated by frequency bins
              const t = (x / tableSize) * Math.PI * 2;
              
              // Map to frequency bins (like iota_instantiation.comp)
              const freqMag = Math.abs((x / tableSize - 0.5) * 2);
              const binIndex = Math.floor(freqMag * 10);
              const bin = Math.min(9, binIndex);
              
              // Each bin has different harmonic coefficient
              const harmonicCoeff = 0.5 + Math.sin(bin * 0.5) * 0.3;
              
              // Generate waveform with bin-specific modulation
              const baseFreq = 1.5;
              let waveform = Math.sin(t * baseFreq) * harmonicCoeff;
              waveform += Math.sin(t * baseFreq * 2) * harmonicCoeff * 0.3;
              waveform += Math.sin(t * baseFreq * 3) * harmonicCoeff * 0.15;
              
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 1.5 + Math.cos(y * 0.015) * 0.4;
            } else if (filterIndex === 2) {
              // Tau: n → 𝟙 (encoding) - Template-aware normalization
              // Brainwave pattern: normalized/projected waveform
              const t = (x / tableSize) * Math.PI * 2;
              
              // Generate base waveform
              let waveform = Math.sin(t * 2.5) * 0.6;
              waveform += Math.sin(t * 5.0) * 0.2;
              
              // Apply normalization (template-aware reduction)
              const templateMag = 0.7;
              const projectionStrength = 0.8;
              const normalized = waveform / (templateMag + 0.001);
              waveform = waveform * (1.0 - projectionStrength) + normalized * templateMag * projectionStrength;
              
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 2.0 + Math.sin(y * 0.012) * 0.4;
            } else if (filterIndex === 3) {
              // Epsilon: 𝟙 → ∞ (evaluation) - Aggregated metrics
              // Brainwave pattern: smoother, aggregated signal (metrics-based)
              const t = (x / tableSize) * Math.PI * 2;
              
              // Generate smoother waveform (lower frequency, more aggregated)
              let waveform = Math.sin(t * 0.8) * 0.5; // Lower frequency
              waveform += Math.sin(t * 1.6) * 0.2;
              
              // Apply metric-like smoothing (energy + coherence)
              const energy = waveform * waveform;
              const coherence = 0.7 + waveform * 0.3;
              waveform = (energy * 0.4 + coherence * 0.6);
              
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 0.8 + Math.cos(y * 0.008) * 0.3;
            } else if (filterIndex === 4) {
              // Proto-Identity (𝟙): Additive combination of γ + ε
              // Brainwave pattern: compound waveform from gamma + epsilon
              const t = (x / tableSize) * Math.PI * 2;
              
              // Gamma component: multi-harmonic with envelope
              let gammaWave = Math.sin(t * 2.0) * 0.3;
              for (let h = 2; h <= 6; h++) {
                gammaWave += Math.pow(0.75, h - 2) * 0.2 * Math.sin(t * 2.0 * h);
              }
              const freqMag = Math.abs((x / tableSize - 0.5) * 2);
              const envelope = Math.exp(-freqMag * freqMag / (2 * 0.4 * 0.4));
              gammaWave *= envelope;
              
              // Epsilon component: smoother aggregated signal
              let epsilonWave = Math.sin(t * 0.8) * 0.3;
              epsilonWave += Math.sin(t * 1.6) * 0.15;
              const epsilonEnergy = epsilonWave * epsilonWave;
              const epsilonCoherence = 0.7 + epsilonWave * 0.3;
              epsilonWave = (epsilonEnergy * 0.4 + epsilonCoherence * 0.6) * 0.4;
              
              // Additive combination - shows compounding
              let waveform = gammaWave + epsilonWave;
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 1.2 + Math.sin((x + y) * 0.01) * 0.3;
            } else {
              // Instances (n): Additive combination of τ + ι
              // Brainwave pattern: compound waveform from tau + iota
              const t = (x / tableSize) * Math.PI * 2;
              
              // Tau component: normalized/projected waveform
              let tauWave = Math.sin(t * 2.5) * 0.4;
              tauWave += Math.sin(t * 5.0) * 0.15;
              const templateMag = 0.7;
              const projectionStrength = 0.8;
              const normalized = tauWave / (templateMag + 0.001);
              tauWave = tauWave * (1.0 - projectionStrength) + normalized * templateMag * projectionStrength;
              
              // Iota component: frequency-bin modulated
              const freqMag = Math.abs((x / tableSize - 0.5) * 2);
              const bin = Math.min(9, Math.floor(freqMag * 10));
              const harmonicCoeff = 0.5 + Math.sin(bin * 0.5) * 0.3;
              let iotaWave = Math.sin(t * 1.5) * harmonicCoeff * 0.4;
              iotaWave += Math.sin(t * 3.0) * harmonicCoeff * 0.2;
              
              // Additive combination - shows compounding
              let waveform = tauWave + iotaWave;
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 2.5 + Math.sin(y * 0.02) * 0.4;
            }
          } else {
            const t = x / tableSize;
            const rowT = y / tableSize;
            
            if (filterIndex === 0) {
              // Gamma: Multi-harmonic brainwave pattern (synthetic)
              const tNorm = t * Math.PI * 2;
              let waveform = Math.sin(tNorm * 2.0) * 0.4;
              for (let h = 2; h <= 8; h++) {
                waveform += Math.pow(0.75, h - 2) * 0.3 * Math.sin(tNorm * 2.0 * h);
              }
              const freqMag = Math.abs(t - 0.5) * 2;
              const envelope = Math.exp(-freqMag * freqMag / (2 * 0.4 * 0.4));
              waveform *= envelope;
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 1.0 + Math.sin(rowT * Math.PI * 2) * 0.3;
            } else if (filterIndex === 1) {
              // Iota: Frequency-bin modulated brainwave (synthetic)
              const tNorm = t * Math.PI * 2;
              const bin = Math.floor(t * 10) % 10;
              const harmonicCoeff = 0.5 + Math.sin(bin * 0.5) * 0.3;
              let waveform = Math.sin(tNorm * 1.5) * harmonicCoeff;
              waveform += Math.sin(tNorm * 3.0) * harmonicCoeff * 0.3;
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 1.5 + Math.cos(rowT * Math.PI * 2) * 0.4;
            } else if (filterIndex === 2) {
              // Tau: Normalized/projected brainwave (synthetic)
              const tNorm = t * Math.PI * 2;
              let waveform = Math.sin(tNorm * 2.5) * 0.5;
              waveform += Math.sin(tNorm * 5.0) * 0.2;
              // Normalize
              waveform *= 0.9;
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 2.0 + Math.sin(rowT * Math.PI * 3) * 0.4;
            } else if (filterIndex === 3) {
              // Epsilon: Aggregated metrics brainwave (synthetic)
              const tNorm = t * Math.PI * 2;
              let waveform = Math.sin(tNorm * 0.8) * 0.4;
              waveform += Math.sin(tNorm * 1.6) * 0.2;
              const energy = waveform * waveform;
              const coherence = 0.7 + waveform * 0.3;
              waveform = (energy * 0.4 + coherence * 0.6);
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 0.8 + Math.cos(rowT * Math.PI * 4) * 0.3;
            } else if (filterIndex === 4) {
              // Proto-Identity (𝟙): Additive γ + ε brainwave (synthetic)
              const tNorm = t * Math.PI * 2;
              
              // Gamma component
              let gammaWave = Math.sin(tNorm * 2.0) * 0.3;
              for (let h = 2; h <= 6; h++) {
                gammaWave += Math.pow(0.75, h - 2) * 0.2 * Math.sin(tNorm * 2.0 * h);
              }
              const freqMag = Math.abs(t - 0.5) * 2;
              const envelope = Math.exp(-freqMag * freqMag / (2 * 0.4 * 0.4));
              gammaWave *= envelope;
              
              // Epsilon component
              let epsilonWave = Math.sin(tNorm * 0.8) * 0.3;
              epsilonWave += Math.sin(tNorm * 1.6) * 0.15;
              const epsilonEnergy = epsilonWave * epsilonWave;
              const epsilonCoherence = 0.7 + epsilonWave * 0.3;
              epsilonWave = (epsilonEnergy * 0.4 + epsilonCoherence * 0.6) * 0.4;
              
              // Additive combination
              let waveform = gammaWave + epsilonWave;
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 1.2 + Math.sin((t + rowT) * Math.PI * 2) * 0.3;
            } else {
              // Instances (n): Additive τ + ι brainwave (synthetic)
              const tNorm = t * Math.PI * 2;
              
              // Tau component
              let tauWave = Math.sin(tNorm * 2.5) * 0.4;
              tauWave += Math.sin(tNorm * 5.0) * 0.15;
              tauWave *= 0.9;
              
              // Iota component
              const bin = Math.floor(t * 10) % 10;
              const harmonicCoeff = 0.5 + Math.sin(bin * 0.5) * 0.3;
              let iotaWave = Math.sin(tNorm * 1.5) * harmonicCoeff * 0.4;
              iotaWave += Math.sin(tNorm * 3.0) * harmonicCoeff * 0.2;
              
              // Additive combination
              let waveform = tauWave + iotaWave;
              intensity = 0.5 + waveform * 0.5;
              intensity = Math.max(0, Math.min(1, intensity));
              frequency = 2.5 + Math.sin(rowT * Math.PI * 5) * 0.4;
            }
          }
        }
        
        intensityData[idx] = Math.max(0, Math.min(1, intensity));
        frequencyData[idx] = Math.max(0.1, Math.min(5.0, frequency));
      }
    }
    
    // Create textures from data with proper filtering for smooth interpolation
    const intensityTexture = new THREE.DataTexture(
      intensityData,
      tableSize,
      tableSize,
      THREE.RedFormat,
      THREE.FloatType
    );
    intensityTexture.needsUpdate = true;
    intensityTexture.minFilter = THREE.LinearFilter; // Bilinear minification
    intensityTexture.magFilter = THREE.LinearFilter; // Bilinear magnification
    intensityTexture.wrapS = THREE.ClampToEdgeWrapping;
    intensityTexture.wrapT = THREE.ClampToEdgeWrapping;
    intensityTexture.generateMipmaps = false; // Disable mipmaps for FloatType
    
    const frequencyTexture = new THREE.DataTexture(
      frequencyData,
      tableSize,
      tableSize,
      THREE.RedFormat,
      THREE.FloatType
    );
    frequencyTexture.needsUpdate = true;
    frequencyTexture.minFilter = THREE.LinearFilter; // Bilinear minification
    frequencyTexture.magFilter = THREE.LinearFilter; // Bilinear magnification
    frequencyTexture.wrapS = THREE.ClampToEdgeWrapping;
    frequencyTexture.wrapT = THREE.ClampToEdgeWrapping;
    frequencyTexture.generateMipmaps = false; // Disable mipmaps for FloatType
    
    // Create a simple plane geometry (just 4 vertices, positions computed in shader)
    const geometry = new THREE.PlaneGeometry(scale, scale, tableSize - 1, tableSize - 1);
    
    // Create shader material that computes everything in the shader
    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        amplitudeScale: { value: 2.0 },
        tableSize: { value: tableSize },
        scale: { value: scale },
        intensityTexture: { value: intensityTexture },
        frequencyTexture: { value: frequencyTexture },
        colorTexture: { value: null } // We'll compute color from row in shader
      },
      vertexShader: `
        uniform float time;
        uniform float amplitudeScale;
        uniform float tableSize;
        uniform float scale;
        uniform sampler2D intensityTexture;
        uniform sampler2D frequencyTexture;
        
        varying vec3 vColor;
        varying float vIntensity;
        varying vec2 vUv;
        
        void main() {
          vUv = uv;
          
          // Compute grid position from UV (0-1 maps to 0-tableSize)
          vec2 gridPos = uv * tableSize;
          
          // Sample intensity (w) and frequency from textures
          // Use sub-pixel coordinates for proper bilinear interpolation in both X and Y
          // This ensures smooth transitions in both directions
          vec2 texCoord = vec2((gridPos.x + 0.5) / tableSize, (gridPos.y + 0.5) / tableSize);
          
          // Sample with automatic bilinear filtering (smooth in both X and Y)
          // LINEAR filter mode provides smooth interpolation automatically
          float intensity = texture2D(intensityTexture, texCoord).r;
          float frequency = texture2D(frequencyTexture, texCoord).r;
          
          vIntensity = intensity;
          
          // Color palette: Orange/Gold for active waveform only
          // All active filters use orange/gold gradient
          vec3 rowColor = vec3(0.0, 0.0, 0.0); // Default black for inactive
          if (intensity > 0.01) {
            // Orange/Gold gradient based on intensity
            float goldFactor = intensity;
            rowColor = vec3(1.0, 0.5 + goldFactor * 0.3, 0.0); // Orange to gold
          }
          vColor = rowColor;
          
          // Wavetable structure: X is time/samples, Z is filter row, Y is amplitude
          // Each row (Z) represents a different filter/waveform
          // Each column (X) represents a time sample
          float baseX = (uv.x - 0.5) * scale; // Time axis (horizontal)
          float baseZ = (uv.y - 0.5) * scale; // Filter row axis (depth)
          
          // Each filter/row moves independently with different speeds and phases
          int filterIndex = int(mod(gridPos.y, 6.0));
          float speedX, speedZ, phaseX, phaseZ;
          
          // Different movement parameters for each filter type (10x faster)
          if (filterIndex == 0) {
            speedX = 3.0; speedZ = 2.0; phaseX = 0.0; phaseZ = 0.0;
          } else if (filterIndex == 1) {
            speedX = 5.0; speedZ = 4.0; phaseX = 1.0; phaseZ = 0.5;
          } else if (filterIndex == 2) {
            speedX = 4.0; speedZ = 3.0; phaseX = 2.0; phaseZ = 1.0;
          } else if (filterIndex == 3) {
            speedX = 6.0; speedZ = 5.0; phaseX = 1.5; phaseZ = 1.5;
          } else if (filterIndex == 4) {
            speedX = 3.5; speedZ = 2.5; phaseX = 0.5; phaseZ = 2.0;
          } else {
            speedX = 4.5; speedZ = 3.5; phaseX = 2.5; phaseZ = 0.75;
          }
          
          // Add position-based variation so even within same filter, waves move differently
          float posVariationX = gridPos.x * 0.01 + gridPos.y * 0.02;
          float posVariationZ = gridPos.y * 0.015 + gridPos.x * 0.025;
          
          // Move waveforms across X and Z axes independently
          float posX = baseX + sin(time * speedX + phaseX + baseZ * 0.1 + posVariationX) * 2.0;
          float posZ = baseZ + cos(time * speedZ + phaseZ + baseX * 0.1 + posVariationZ) * 2.0;
          
          // Y position is the waveform amplitude (height)
          // Add subtle animation for "living" brainwave effect - 10x faster
          float pulse = sin(time * frequency * 5.0 + posX * 0.2) * 0.1 + 1.0;
          float animatedY = intensity * amplitudeScale * pulse;
          
          // Offset each filter row vertically to show wavetable structure
          float rowOffset = (mod(gridPos.y, 6.0) - 2.5) * 0.5; // Stack rows
          animatedY += rowOffset;
          
          // XZ plane, Y is up
          vec3 pos = vec3(posX, animatedY, posZ);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vIntensity;
        varying vec2 vUv;
        
        void main() {
          // For additive blending, transparency comes from reducing color intensity, not alpha
          // Scale down color values significantly to make it more transparent
          float transparencyFactor = 0.12; // Much more transparent (12% intensity)
          vec3 finalColor = vColor * (0.5 + vIntensity * 0.5) * transparencyFactor;
          
          // With additive blending, alpha is used for masking but color intensity controls brightness
          float alpha = 0.4 + vIntensity * 0.3; // Alpha for smooth edges
          gl_FragColor = vec4(finalColor, alpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending, // Explicitly use additive blending
      side: THREE.DoubleSide,
      wireframe: false
    });
    
    wavetableLines = new THREE.Mesh(geometry, material);
    wavetableLines.position.set(0, 0, 0);
    wavetableLines.rotation.y = Math.PI / 2; // Rotate 90 degrees around Y axis
    scene.add(wavetableLines);
    
    console.log(`✅ Wavetable plane created (shader-based): ${tableSize}x${tableSize} grid`);
  }
  
  function getFilterColor(row) {
    if (row === 0) return [1.0, 0.3, 0.0]; // Gamma - orange
    if (row === 1) return [0.0, 1.0, 0.5]; // Iota - cyan-green
    if (row === 2) return [0.5, 0.0, 1.0]; // Tau - purple
    if (row === 3) return [1.0, 0.8, 0.0]; // Epsilon - yellow
    if (row === 4) return [0.0, 0.8, 1.0]; // Proto-Identity - blue
    return [1.0, 0.5, 0.8]; // Instances - pink
  }
  
  // Reactive updates - debounced to prevent constant flickering
  let updateTimeout;
  $: {
    if (scene && particleSystem !== undefined) {
      clearTimeout(updateTimeout);
      updateTimeout = setTimeout(() => {
        updateParticles();
      }, 100); // Debounce updates
    }
  }
  
  $: {
    if (scene && wavetableLines !== undefined) {
      clearTimeout(updateTimeout);
      updateTimeout = setTimeout(() => {
        updateWavetable();
      }, 100); // Debounce updates
    }
  }
  
  let animationFrame;
  let time = 0;
  
  function animate() {
    animationFrame = requestAnimationFrame(animate);
    time += 0.016; // ~60fps
    
    if (particleSystem && particleSystem.material.uniforms) {
      particleSystem.material.uniforms.time.value = time;
    }
    
    // Animate wavetable oscillation
    if (wavetableLines && wavetableLines.material.uniforms) {
      wavetableLines.material.uniforms.time.value = time;
    }
    
    if (controls) {
      controls.update();
    }
    
    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
  }
  
  onMount(async () => {
    console.log('🚀 BrainSpace.svelte mounting with Three.js...');
    
    // Subscribe to filter state
    filterUnsubscribe = filterState.subscribe(value => {
      selectedFilters = value;
      console.log('🔄 Filter state updated:', selectedFilters);
      updateWavetable();
    });
    
    if (!canvasContainer) {
      console.error('❌ Canvas container not found!');
      return;
    }
    
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000011); // Deep navy/black background
    
    // Create camera - Portrait view (looking forward at brain)
    camera = new THREE.PerspectiveCamera(
      45, // FOV
      canvasContainer.clientWidth / canvasContainer.clientHeight,
      0.1,
      200
    );
    // Portrait view: camera in front, looking at brain (not from above)
    camera.position.set(0, 0, 30); // In front of the brain
    camera.lookAt(0, 0, 0); // Look at center
    
    // Create renderer
    const existingCanvas = canvasContainer.querySelector('canvas');
    renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true,
      canvas: existingCanvas || undefined
    });
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    if (!existingCanvas) {
      canvasContainer.appendChild(renderer.domElement);
    }
    
    // Add OrbitControls for camera interaction (optional - will work without it)
    try {
      // Try to use OrbitControls if available
      const OrbitControlsModule = await import('three/examples/jsm/controls/OrbitControls.js');
      const { OrbitControls } = OrbitControlsModule;
      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.minDistance = 2;
      controls.maxDistance = 50;
      console.log('✅ OrbitControls initialized');
    } catch (err) {
      console.warn('⚠️  OrbitControls not available, using basic camera. Install with: npm install three');
      // Basic mouse controls
      let isDragging = false;
      let lastMouseX = 0;
      let lastMouseY = 0;
      const handleMouseDown = (e) => {
        isDragging = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
      };
      const handleMouseMove = (e) => {
        if (!isDragging) return;
        const deltaX = e.clientX - lastMouseX;
        const deltaY = e.clientY - lastMouseY;
        camera.rotation.y += deltaX * 0.01;
        camera.rotation.x += deltaY * 0.01;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
      };
      const handleMouseUp = () => { isDragging = false; };
      const handleWheel = (e) => {
        e.preventDefault();
        camera.position.z += e.deltaY * 0.01;
        camera.position.z = Math.max(2, Math.min(50, camera.position.z));
      };
      renderer.domElement.addEventListener('mousedown', handleMouseDown);
      renderer.domElement.addEventListener('mousemove', handleMouseMove);
      renderer.domElement.addEventListener('mouseup', handleMouseUp);
      renderer.domElement.addEventListener('mouseleave', handleMouseUp);
      renderer.domElement.addEventListener('wheel', handleWheel);
    }
    
    // Handle resize
    const handleResize = () => {
      if (!canvasContainer || !camera || !renderer) return;
      const width = canvasContainer.clientWidth;
      const height = canvasContainer.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);
    
    console.log('✅ Three.js scene initialized');
    
    // Initial updates
    setTimeout(() => {
      updateParticles();
      updateWavetable();
    }, 100);
    
    // Start animation loop
    animate();
    
    return () => {
      console.log('🧹 BrainSpace.svelte cleaning up...');
      if (filterUnsubscribe) {
        filterUnsubscribe();
      }
      window.removeEventListener('resize', handleResize);
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
      if (particleSystem) {
        particleSystem.geometry.dispose();
        particleSystem.material.dispose();
      }
      if (wavetableLines) {
        wavetableLines.traverse(child => {
          if (child.geometry) child.geometry.dispose();
          if (child.material) child.material.dispose();
        });
      }
      if (renderer) {
        renderer.dispose();
      }
    };
  });
</script>

<div bind:this={canvasContainer} class="absolute inset-0" style="z-index: 0;">
  <canvas></canvas>
</div>

<style>
  :global(canvas) {
    display: block;
    width: 100%;
    height: 100%;
  }
</style>
