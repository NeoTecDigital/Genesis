<script>
  import { onMount, onDestroy } from 'svelte';
  
  export let activation = { waveform: [], sparks: [] };
  export let gl;
  
  let program;
  let waveformBuffer;
  let sparkBuffer;
  
  // Vertex shader for waveform
  const vertexShaderSource = `#version 300 es
    in vec2 a_position;
    in float a_amplitude;
    in float a_phase;
    
    uniform mat4 u_projection;
    uniform float u_time;
    uniform float u_activation_energy;
    
    out vec3 v_color;
    out float v_energy;
    
    void main() {
      // Waveform position (horizontal line through center)
      float y = 0.0 + sin(a_phase + u_time * 2.0) * a_amplitude * 0.1;
      gl_Position = u_projection * vec4(a_position.x, y, 0.0, 1.0);
      
      // Color transition: cyan (idle) to orange (active)
      vec3 cyan = vec3(0.0, 0.8, 1.0);
      vec3 orange = vec3(1.0, 0.5, 0.0);
      v_color = mix(cyan, orange, u_activation_energy);
      v_energy = a_amplitude;
    }
  `;
  
  // Fragment shader
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    
    in vec3 v_color;
    in float v_energy;
    
    out vec4 fragColor;
    
    void main() {
      // Glowing waveform line
      float glow = exp(-v_energy * 2.0) * 0.5;
      vec3 color = v_color + vec3(glow);
      fragColor = vec4(color, 1.0);
    }
  `;
  
  // Spark particle shader
  const sparkVertexShader = `#version 300 es
    in vec2 a_position;
    in float a_intensity;
    
    uniform mat4 u_projection;
    uniform float u_time;
    
    out float v_intensity;
    
    void main() {
      // Animate sparks (expand outward)
      vec2 pos = a_position * (1.0 + u_time * 0.5);
      gl_Position = u_projection * vec4(pos, 0.0, 1.0);
      gl_PointSize = a_intensity * 10.0;
      v_intensity = a_intensity;
    }
  `;
  
  const sparkFragmentShader = `#version 300 es
    precision highp float;
    
    in float v_intensity;
    
    out vec4 fragColor;
    
    void main() {
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord);
      
      // Orange spark with glow
      float alpha = (1.0 - smoothstep(0.0, 0.5, dist)) * v_intensity;
      vec3 color = vec3(1.0, 0.5, 0.0);
      fragColor = vec4(color, alpha);
    }
  `;
  
  function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    
    return shader;
  }
  
  function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      return null;
    }
    
    return program;
  }
  
  function updateWaveform() {
    if (!gl || !program || !activation.waveform || activation.waveform.length === 0) return;
    
    const waveform = activation.waveform;
    const positions = new Float32Array(waveform.length * 2);
    const amplitudes = new Float32Array(waveform.length);
    const phases = new Float32Array(waveform.length);
    
    const canvas = gl.canvas;
    const width = canvas.width;
    const height = canvas.height;
    
    for (let i = 0; i < waveform.length; i++) {
      const x = (i / waveform.length) * 2 - 1; // -1 to 1
      positions[i * 2] = x;
      positions[i * 2 + 1] = 0; // Will be animated in shader
      amplitudes[i] = waveform[i] || 0;
      phases[i] = i * 0.1;
    }
    
    gl.bindBuffer(gl.ARRAY_BUFFER, waveformBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
  }
  
  function setup() {
    if (!gl) return;
    
    // Create waveform program
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    
    if (vertexShader && fragmentShader) {
      program = createProgram(gl, vertexShader, fragmentShader);
    }
    
    waveformBuffer = gl.createBuffer();
    
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  }
  
  function render(time) {
    if (!gl || !program || !activation.waveform || activation.waveform.length === 0) return;
    
    gl.useProgram(program);
    
    // Set up projection
    const canvas = gl.canvas;
    const projection = new Float32Array([
      2 / canvas.width, 0, 0, 0,
      0, 2 / canvas.height, 0, 0,
      0, 0, -1, 0,
      -1, -1, 0, 1
    ]);
    
    const projLoc = gl.getUniformLocation(program, 'u_projection');
    const timeLoc = gl.getUniformLocation(program, 'u_time');
    const energyLoc = gl.getUniformLocation(program, 'u_activation_energy');
    
    gl.uniformMatrix4fv(projLoc, false, projection);
    gl.uniform1f(timeLoc, time * 0.001);
    gl.uniform1f(energyLoc, activation.energy || 0.5);
    
    // Draw waveform
    gl.bindBuffer(gl.ARRAY_BUFFER, waveformBuffer);
    const posLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    
    gl.drawArrays(gl.LINE_STRIP, 0, activation.waveform.length);
  }
  
  $: if (activation.waveform && activation.waveform.length > 0 && gl && program) {
    updateWaveform();
  }
  
  onMount(() => {
    setup();
    
    let animationFrame;
    const loop = (time) => {
      render(time);
      animationFrame = requestAnimationFrame(loop);
    };
    animationFrame = requestAnimationFrame(loop);
    
    return () => {
      if (animationFrame) cancelAnimationFrame(animationFrame);
    };
  });
</script>

