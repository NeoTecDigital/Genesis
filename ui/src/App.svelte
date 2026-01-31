<script>
  import { onMount } from 'svelte';
  import BrainSpace from './components/BrainSpace.svelte';
  import ControlPanel from './components/ControlPanel.svelte';
  import ChatBar from './components/ChatBar.svelte';
  import { websocketStore } from './stores/websocket.js';
  
  onMount(() => {
    console.log('🎨 App.svelte mounted');
    
    // Initialize WebSocket connection (disabled for now - using mock data)
    // websocketStore.connect('ws://localhost:8000/ws');
    
    console.log('📊 Setting mock data...');
    // Generate realistic memory brain data at scale
    // Each cluster represents a proto-identity with multiple memories
    const numClusters = 1000; // Number of proto-identities
    const clusters = [];
    const allMemories = []; // Flat list of all memory points for point cloud
    
    for (let i = 0; i < numClusters; i++) {
      // Each cluster has a centroid (proto-identity location)
      const centroid = [
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 20
      ];
      
      // Each cluster contains multiple memories (memory_ids)
      // Realistic distribution: some clusters have many memories, some have few
      const memoryCount = Math.floor(Math.random() * 50) + 5; // 5-55 memories per cluster
      const memoryIds = Array.from({ length: memoryCount }, (_, j) => allMemories.length + j);
      
      // Generate memory points around the cluster centroid
      // Memories cluster around their proto-identity with some variance
      for (let m = 0; m < memoryCount; m++) {
        // Gaussian-like distribution around centroid
        const angle1 = Math.random() * Math.PI * 2;
        const angle2 = Math.random() * Math.PI;
        const radius = Math.random() * 2.0 + 0.5; // 0.5-2.5 units from centroid
        
        const memoryPos = [
          centroid[0] + Math.cos(angle1) * Math.sin(angle2) * radius,
          centroid[1] + Math.sin(angle1) * Math.sin(angle2) * radius,
          centroid[2] + Math.cos(angle2) * radius
        ];
        
        allMemories.push({
          memory_id: memoryIds[m],
          cluster_id: i,
          identity_id: i % 5, // 5 different identity types
          position: memoryPos,
          coherence: 0.7 + Math.random() * 0.3, // 0.7-1.0
          timestamp: Date.now() - Math.random() * 1000000
        });
      }
      
      clusters.push({
        cluster_id: i,
        identity_id: i % 5,
        centroid: centroid,
        memory_ids: memoryIds,
        memory_count: memoryCount,
        coherence: 0.8 + Math.random() * 0.2, // Cluster coherence
        metadata: {
          created_at: Date.now() - Math.random() * 2000000,
          access_count: Math.floor(Math.random() * 100)
        }
      });
    }
    
    console.log(`📦 Generated ${clusters.length} clusters with ${allMemories.length} total memories`);
    
    const mockData = {
      brainSpace: {
        clusters: clusters,
        memories: allMemories // Full point cloud data
      },
      activation: {
        waveform: Array.from({ length: 512 }, (_, i) => 
          Math.sin(i * 0.1) * (1 - Math.abs(i - 256) / 256) * 0.5
        ),
        sparks: Array.from({ length: 5 }, () => ({
          x: Math.random() * 512,
          y: Math.random() * 512,
          intensity: Math.random()
        })),
        energy: 0.7,
        morphism: 'gamma'
      },
      controls: {
        current_cycle: 'active',
        morphism_state: 'gamma',
        coherence: 0.85,
        gamma_params: { amplitude: 100.0 },
        tau_params: { projection_strength: 0.8 }
      },
      events: []
    };
    
    websocketStore.data.set(mockData);
    websocketStore.connected.set(true);
    console.log('✅ Mock data set, connected state: true');
    console.log('📦 Mock clusters:', mockData.brainSpace.clusters.length);
  });
</script>

<main class="h-screen w-screen flex flex-col relative">
  <!-- Main 3D Brain Space -->
  <div class="flex-1 relative">
    <BrainSpace />
  </div>
  
  <!-- Control Panel (Right Sidebar) -->
  <ControlPanel />
  
  <!-- Chat Bar (Bottom) -->
  <ChatBar />
</main>

<style>
  main {
    background: #000011; /* Deep navy/black background */
    position: relative;
  }
  
  /* Vignette effect - radial gradient overlay (behind UI) */
  main::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(
      ellipse at center,
      transparent 0%,
      transparent 40%,
      rgba(0, 0, 17, 0.3) 70%,
      rgba(0, 0, 0, 0.8) 100%
    );
    pointer-events: none;
    z-index: 0;
  }
  
  /* Additional radial filter for depth */
  main::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(
      circle at 50% 50%,
      transparent 0%,
      rgba(0, 17, 34, 0.2) 50%,
      rgba(0, 0, 0, 0.6) 100%
    );
    pointer-events: none;
    z-index: 0;
  }
</style>

