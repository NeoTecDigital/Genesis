<script>
  export let active = false;
</script>

<button
  class="px-3 py-1.5 text-xs rounded transition-colors {active 
    ? 'bg-genesis-cyan text-black' 
    : 'bg-white/10 text-gray-300 hover:bg-white/20'}"
  on:click
>
  <slot />
</button>

