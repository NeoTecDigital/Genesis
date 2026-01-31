<script>
  export let value = false;
  
  function toggle() {
    value = !value;
    dispatch('toggle', value);
  }
  
  import { createEventDispatcher } from 'svelte';
  const dispatch = createEventDispatcher();
</script>

<button
  on:click={toggle}
  class="relative w-12 h-6 rounded-full transition-colors {value 
    ? 'bg-genesis-cyan' 
    : 'bg-gray-600'}"
>
  <span
    class="absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform {value 
      ? 'translate-x-6' 
      : 'translate-x-0'}"
  />
</button>

