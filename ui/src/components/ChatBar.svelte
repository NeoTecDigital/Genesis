<script>
  import { onMount } from 'svelte';
  import { websocketStore } from '../stores/websocket.js';
  
  onMount(() => {
    console.log('💬 ChatBar.svelte mounted');
  });
  
  let input = '';
  let history = [];
  
  function handleSubmit() {
    if (!input.trim()) {
      console.log('⚠️  Empty command, ignoring');
      return;
    }
    
    console.log('📤 Sending command:', input);
    
    // Add to history
    history = [...history, { type: 'user', message: input }];
    console.log('📝 History updated, count:', history.length);
    
    // Send command
    websocketStore.send({
      type: 'chat_command',
      command: input
    });
    
    input = '';
  }
  
  function handleKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  }
</script>

<!-- Floating Capsule Chat Bar -->
<div class="chat-capsule">
  <div class="capsule-content">
    <!-- Left: Image Upload Icon -->
    <button
      type="button"
      class="attach-button"
      title="Add Image/Context"
      on:click={() => console.log('📎 Image upload clicked')}
    >
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
      </svg>
    </button>
    
    <!-- Center: Input Field -->
    <input
      type="text"
      bind:value={input}
      on:keydown={handleKeydown}
      placeholder="Type command: recall [x,y,z], generate, optimize..."
      class="chat-input"
    />
    
    <!-- Right: Play Audio Button -->
    <button
      type="button"
      class="play-button"
      title="Play Audio"
      on:click={() => console.log('🔊 Play audio clicked')}
    >
      <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M8 5v14l11-7z"/>
      </svg>
    </button>
  </div>
  
  {#if history.length > 0}
    <div class="history-preview">
      {#each history.slice(-3) as item}
        <div class="history-item">
          <span class="text-genesis-cyan">&gt;</span> {item.message}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .chat-capsule {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 20;
    width: 90%;
    max-width: 600px;
    margin: 0 auto;
  }
  
  .capsule-content {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(10, 20, 40, 0.6);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 255, 255, 0.3);
    border-radius: 999px;
    padding: 8px 16px;
    box-shadow: 0 4px 20px rgba(0, 255, 255, 0.1);
  }
  
  .attach-button {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    color: rgba(0, 255, 255, 0.7);
    background: transparent;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    transition: all 0.2s;
    flex-shrink: 0;
  }
  
  .attach-button:hover {
    color: rgba(0, 255, 255, 1);
    background: rgba(0, 255, 255, 0.1);
  }
  
  .chat-input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: rgba(255, 255, 255, 0.9);
    font-size: 14px;
    padding: 4px 8px;
  }
  
  .chat-input::placeholder {
    color: rgba(255, 255, 255, 0.4);
  }
  
  .play-button {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    color: rgba(255, 165, 0, 0.9);
    background: rgba(255, 165, 0, 0.1);
    border: 1px solid rgba(255, 165, 0, 0.3);
    border-radius: 50%;
    cursor: pointer;
    transition: all 0.2s;
    flex-shrink: 0;
    box-shadow: 0 0 10px rgba(255, 165, 0, 0.3);
  }
  
  .play-button:hover {
    color: rgba(255, 165, 0, 1);
    background: rgba(255, 165, 0, 0.2);
    box-shadow: 0 0 15px rgba(255, 165, 0, 0.5);
  }
  
  .history-preview {
    margin-top: 8px;
    padding: 8px 16px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    max-height: 80px;
    overflow-y: auto;
  }
  
  .history-item {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.6);
    margin-bottom: 4px;
  }
</style>

