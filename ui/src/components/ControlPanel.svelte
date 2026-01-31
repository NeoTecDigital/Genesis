<script>
  import { onMount, onDestroy } from 'svelte';
  import { websocketStore, dataStore, connectedStore } from '../stores/websocket.js';
  import Button from './ui/Button.svelte';
  import Toggle from './ui/Toggle.svelte';
  import Slider from './ui/Slider.svelte';
  
  onMount(() => {
    console.log('🎛️  ControlPanel.svelte mounted');
  });
  
  let activeTab = 'monitors';
  let filterThreshold = 0.95;
  let gpuEnabled = true;
  let isOpen = true;
  
  import { filterState } from '../stores/filters.js';
  
  // Filter selection state - sync with store
  let selectedFilters = {
    gamma: true,
    iota: true,
    tau: true,
    epsilon: true,
    protoIdentity: true,
    instances: true
  };
  
  let filterUnsubscribe;
  
  onMount(() => {
    // Subscribe to store updates
    filterUnsubscribe = filterState.subscribe(value => {
      selectedFilters = value;
    });
  });
  
  onDestroy(() => {
    if (filterUnsubscribe) filterUnsubscribe();
  });
  
  // Update store when local state changes
  function updateFilter(filterName, value) {
    selectedFilters[filterName] = value;
    filterState.set(selectedFilters);
  }
  
  $: controls = $dataStore?.controls || {};
  $: events = $dataStore?.events || [];
  $: connected = $connectedStore;
  
  $: {
    console.log('🔄 ControlPanel data updated:', {
      activeTab,
      controls: Object.keys(controls || {}),
      eventsCount: events?.length || 0,
      connected
    });
  }
  
  function handleCommand(command) {
    websocketStore.send({ type: 'command', command });
  }
  
  function updateThreshold(value) {
    filterThreshold = value;
    websocketStore.send({ 
      type: 'update_setting', 
      setting: 'coherence_threshold', 
      value 
    });
  }
  
  function toggleGPU(value) {
    gpuEnabled = value;
    websocketStore.send({ 
      type: 'update_setting', 
      setting: 'gpu_enabled', 
      value 
    });
  }
</script>

<!-- Floating Glassmorphic Sidebar -->
<div class="floating-sidebar {isOpen ? 'open' : 'closed'}" style="width: 320px;">
  <div class="sidebar-content">
  <!-- Connection Status -->
  <div class="mb-4 text-xs">
    <div class="flex items-center gap-2">
      <div class="w-2 h-2 rounded-full {connected ? 'bg-green-500' : 'bg-red-500'}"></div>
      <span class="text-gray-300">{connected ? 'Connected' : 'Disconnected'}</span>
    </div>
  </div>
  
  <!-- Navigation -->
  <nav class="flex gap-2 mb-6">
    <Button 
      active={activeTab === 'logs'} 
      on:click={() => activeTab = 'logs'}
    >
      Logs
    </Button>
    <Button 
      active={activeTab === 'monitors'} 
      on:click={() => activeTab = 'monitors'}
    >
      Monitors
    </Button>
    <Button 
      active={activeTab === 'settings'} 
      on:click={() => activeTab = 'settings'}
    >
      Settings
    </Button>
    <Button 
      active={activeTab === 'help'} 
      on:click={() => activeTab = 'help'}
    >
      Help
    </Button>
  </nav>
  
  <!-- Content -->
  {#if activeTab === 'monitors'}
    <section class="mb-6">
      <h3 class="text-sm font-semibold mb-3 text-genesis-cyan">Categorical Flow</h3>
      <div class="text-xs space-y-2">
        <div>
          <span class="text-gray-400">Current Cycle:</span>
          <span class="ml-2">{controls?.current_cycle || 'idle'}</span>
        </div>
        <div>
          <span class="text-gray-400">Morphism:</span>
          <span class="ml-2 text-genesis-orange">{controls?.morphism_state || 'none'}</span>
        </div>
        <div>
          <span class="text-gray-400">Coherence:</span>
          <span class="ml-2">{controls?.coherence?.toFixed(4) || '0.0000'}</span>
        </div>
      </div>
    </section>
    
    <section class="mb-6">
      <h3 class="text-sm font-semibold mb-3 text-genesis-cyan">Parameter Time Series</h3>
      <div class="text-xs space-y-1 text-gray-300">
        <div>γ amplitude: {controls?.gamma_params?.amplitude?.toFixed(2) || 'N/A'}</div>
        <div>τ strength: {controls?.tau_params?.projection_strength?.toFixed(2) || 'N/A'}</div>
      </div>
    </section>
    
    <section class="mb-6">
      <h3 class="text-sm font-semibold mb-3 text-genesis-cyan">Filters</h3>
      <div class="text-xs space-y-2">
        <label class="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={selectedFilters.gamma} on:change={(e) => updateFilter('gamma', e.target.checked)} class="w-4 h-4" />
          <span class="text-gray-300">Filter Gamma (γ): ∅ → 𝟙</span>
        </label>
        <label class="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={selectedFilters.iota} on:change={(e) => updateFilter('iota', e.target.checked)} class="w-4 h-4" />
          <span class="text-gray-300">Filter Iota (ι): 𝟙 → n</span>
        </label>
        <label class="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={selectedFilters.tau} on:change={(e) => updateFilter('tau', e.target.checked)} class="w-4 h-4" />
          <span class="text-gray-300">Filter Tau (τ): n → 𝟙</span>
        </label>
        <label class="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={selectedFilters.epsilon} on:change={(e) => updateFilter('epsilon', e.target.checked)} class="w-4 h-4" />
          <span class="text-gray-300">Filter Epsilon (ε): 𝟙 → ∞</span>
        </label>
      </div>
    </section>
    
    <section class="mb-6">
      <h3 class="text-sm font-semibold mb-3 text-genesis-cyan">Object Filters</h3>
      <div class="text-xs space-y-2">
        <label class="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={selectedFilters.protoIdentity} on:change={(e) => updateFilter('protoIdentity', e.target.checked)} class="w-4 h-4" />
          <span class="text-gray-300">Filter Proto-Identity (𝟙)</span>
        </label>
        <label class="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={selectedFilters.instances} on:change={(e) => updateFilter('instances', e.target.checked)} class="w-4 h-4" />
          <span class="text-gray-300">Filter Instances (n)</span>
        </label>
      </div>
    </section>
  {/if}
  
  {#if activeTab === 'logs'}
    <section class="space-y-2">
      {#each events.slice(-20).reverse() as event}
        <div class="text-xs p-2 bg-black/20 rounded">
          <div class="text-genesis-cyan">{event.type}</div>
          <div class="text-gray-400">{event.message}</div>
          <div class="text-gray-500 text-[10px]">{event.timestamp}</div>
        </div>
      {/each}
    </section>
  {/if}
  
  {#if activeTab === 'settings'}
    <section class="space-y-4">
      <div>
        <div class="text-sm text-gray-300 block mb-2">GPU Enabled</div>
        <Toggle value={gpuEnabled} on:toggle={toggleGPU} />
      </div>
      
      <div>
        <div class="text-sm text-gray-300 block mb-2">Filter Threshold</div>
        <Slider 
          value={filterThreshold} 
          min={0.0} 
          max={1.0} 
          step={0.01}
          on:change={(e) => updateThreshold(e.detail.value)}
        />
        <div class="text-xs text-gray-400 mt-1">{filterThreshold.toFixed(2)}</div>
      </div>
    </section>
  {/if}
  
  {#if activeTab === 'help'}
    <section class="text-xs space-y-3 text-gray-300">
      <div>
        <h4 class="text-genesis-cyan font-semibold mb-1">Brain Space</h4>
        <p>Cyan particles represent proto-identity clusters in memory. Density indicates coherence.</p>
      </div>
      <div>
        <h4 class="text-genesis-cyan font-semibold mb-1">Waveform</h4>
        <p>Orange waveform shows current activation. Sparks indicate memory recalls.</p>
      </div>
      <div>
        <h4 class="text-genesis-cyan font-semibold mb-1">Commands</h4>
        <p>Use chat bar to send commands: recall, generate, optimize</p>
      </div>
    </section>
  {/if}
  
    <!-- Footer removed -->
  </div>
  
  <!-- Toggle Button -->
  <button
    on:click={() => isOpen = !isOpen}
    class="absolute top-4 {isOpen ? 'left-4' : '-left-12'} transition-all duration-300 z-20 bg-black/50 hover:bg-black/70 text-white p-2 rounded-l-lg border border-white/20 border-r-0"
    title={isOpen ? 'Collapse' : 'Expand'}
  >
    {#if isOpen}
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
      </svg>
    {:else}
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
      </svg>
    {/if}
  </button>
</div>

<style>
  .floating-sidebar {
    position: fixed;
    right: 0;
    top: 0;
    height: 100vh;
    z-index: 15;
    transition: transform 0.3s ease;
  }
  
  .floating-sidebar.open {
    transform: translateX(0);
  }
  
  .floating-sidebar.closed {
    transform: translateX(100%);
  }
  
  .sidebar-content {
    height: 100%;
    width: 100%;
    padding: 24px;
    overflow-y: auto;
    background: rgba(10, 20, 40, 0.6);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    /* Remove borders - use spacing instead */
  }
  
  /* Remove checkbox borders, use spacing */
  :global(.floating-sidebar input[type="checkbox"]) {
    accent-color: rgba(0, 255, 255, 0.8);
    width: 16px;
    height: 16px;
    cursor: pointer;
  }
  
  /* Typography-based separation instead of borders */
  section {
    margin-bottom: 24px;
    padding-bottom: 16px;
  }
  
  section:not(:last-child) {
    border-bottom: none;
  }
</style>

