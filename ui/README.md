# Genesis Visualization Frontend

Svelte-based visualization interface for the Genesis Brain Space system.

## Tech Stack

- **Svelte** (not SvelteKit) - Component framework
- **Routify** - Routing (if needed for multiple views)
- **WebGL** - Direct WebGL for 3D rendering (no Three.js)
- **Tailwind CSS** - Styling (shadcn-inspired components)
- **WebSocket** - Real-time data streaming

## Setup

```bash
cd visualization
npm install
npm run dev
```

## Structure

```
src/
  components/
    BrainSpace.svelte      # Main 3D container
    ControlPanel.svelte    # Right sidebar
    ChatBar.svelte         # Bottom input
    webgl/
      ParticleSystem.svelte    # Brain cluster particles
      ActivationWaveform.svelte # Orange waveform
    ui/
      Button.svelte
      Toggle.svelte
      Slider.svelte
  stores/
    websocket.js           # WebSocket connection store
  App.svelte
  main.js
```

## Backend Integration

The frontend expects a WebSocket server at `ws://localhost:8000/ws` that sends:

```json
{
  "type": "initial_state",
  "brain_space": { "clusters": [...] },
  "controls": { ... }
}
```

See `../src/visualization/server.py` for backend implementation.

