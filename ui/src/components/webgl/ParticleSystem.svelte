<script>
  import { onMount, onDestroy } from 'svelte';
  
  export let clusters = [];
  export let gl;
  
  let program;
  let positionBuffer;
  let colorBuffer;
  let sizeBuffer;
  let vertexCount = 0;
  
  // Vertex shader
  const vertexShaderSource = `#version 300 es
    in vec3 a_position;
    in vec3 a_color;
    in float a_size;
    
    uniform mat4 u_projection;
    uniform mat4 u_view;
    uniform float u_time;
    
    out vec3 v_color;
    out float v_size;
    
    void main() {
      // Animate particles with subtle movement
      vec3 pos = a_position;
      pos.x += sin(u_time * 0.5 + a_position.y * 0.1) * 0.01;
      pos.y += cos(u_time * 0.3 + a_position.x * 0.1) * 0.01;
      
      gl_Position = u_projection * u_view * vec4(pos, 1.0);
      gl_PointSize = a_size * (1.0 + sin(u_time + a_position.z) * 0.2);
      v_color = a_color;
      v_size = a_size;
    }
  `;
  
  // Fragment shader
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    
    in vec3 v_color;
    in float v_size;
    
    out vec4 fragColor;
    
    void main() {
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord);
      
      // Circular particle with glow
      float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
      float glow = exp(-dist * 3.0) * 0.5;
      
      vec3 color = v_color + vec3(glow);
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
  
  function updateBuffers() {
    if (!gl || !program || clusters.length === 0) return;
    
    const positions = new Float32Array(clusters.length * 3);
    const colors = new Float32Array(clusters.length * 3);
    const sizes = new Float32Array(clusters.length);
    
    clusters.forEach((cluster, i) => {
      // Position from centroid (normalize to -1 to 1)
      const pos = cluster.position || cluster.centroid || [0, 0, 0];
      positions[i * 3] = pos[0] * 0.1; // Scale down
      positions[i * 3 + 1] = pos[1] * 0.1;
      positions[i * 3 + 2] = pos[2] * 0.1;
      
      // Color based on coherence (cyan to electric blue)
      const coherence = cluster.coherence || 0.0;
      colors[i * 3] = 0.0; // R
      colors[i * 3 + 1] = 0.5 + coherence * 0.5; // G
      colors[i * 3 + 2] = 1.0; // B
      
      // Size based on memory count
      sizes[i] = 0.02 + (cluster.memory_count || 0) * 0.005;
    });
    
    vertexCount = clusters.length;
    
    // Update position buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    
    // Update color buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
    
    // Update size buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, sizes, gl.DYNAMIC_DRAW);
  }
  
  function setup() {
    if (!gl) return;
    
    // Create shaders
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    
    if (!vertexShader || !fragmentShader) return;
    
    // Create program
    program = createProgram(gl, vertexShader, fragmentShader);
    if (!program) return;
    
    // Create buffers
    positionBuffer = gl.createBuffer();
    colorBuffer = gl.createBuffer();
    sizeBuffer = gl.createBuffer();
    
    // Enable blending for particle glow
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    updateBuffers();
  }
  
  function render(time) {
    if (!gl || !program || vertexCount === 0) return;
    
    // Clear
    gl.clearColor(0.0, 0.0, 0.05, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
    gl.useProgram(program);
    
    // Set up matrices (simple orthographic for now)
    const canvas = gl.canvas;
    const projection = new Float32Array([
      2 / canvas.width, 0, 0, 0,
      0, 2 / canvas.height, 0, 0,
      0, 0, -1, 0,
      -1, -1, 0, 1
    ]);
    
    const view = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ]);
    
    // Set uniforms
    const projLoc = gl.getUniformLocation(program, 'u_projection');
    const viewLoc = gl.getUniformLocation(program, 'u_view');
    const timeLoc = gl.getUniformLocation(program, 'u_time');
    
    gl.uniformMatrix4fv(projLoc, false, projection);
    gl.uniformMatrix4fv(viewLoc, false, view);
    gl.uniform1f(timeLoc, time * 0.001);
    
    // Bind and draw
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    const posLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    const colorLoc = gl.getAttribLocation(program, 'a_color');
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, 0, 0);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuffer);
    const sizeLoc = gl.getAttribLocation(program, 'a_size');
    gl.enableVertexAttribArray(sizeLoc);
    gl.vertexAttribPointer(sizeLoc, 1, gl.FLOAT, false, 0, 0);
    
    gl.drawArrays(gl.POINTS, 0, vertexCount);
  }
  
  $: if (clusters.length > 0 && gl && program) {
    updateBuffers();
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

