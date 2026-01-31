import { writable, derived } from 'svelte/store';

class WebSocketStore {
  constructor() {
    if (WebSocketStore.instance) {
      console.log('🔄 WebSocketStore: Returning existing singleton instance');
      return WebSocketStore.instance;
    }
    
    console.log('🆕 WebSocketStore: Creating new singleton instance');
    this.ws = null;
    this.data = writable({
      brainSpace: { clusters: [], memories: [] },
      activation: { waveform: [], sparks: [] },
      controls: {},
      events: []
    });
    this.connected = writable(false);
    
    WebSocketStore.instance = this;
  }
  
  connect(url) {
    console.log('🔌 WebSocketStore.connect() called with URL:', url);
    try {
      console.log('📡 Creating WebSocket connection...');
      this.ws = new WebSocket(url);
      
      this.ws.onopen = () => {
        console.log('✅ WebSocket connected to:', url);
        this.connected.set(true);
      };
      
      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('📨 WebSocket message received:', message.type, message);
          this.handleMessage(message);
        } catch (error) {
          console.error('❌ Failed to parse WebSocket message:', error, event.data);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        this.connected.set(false);
      };
      
      this.ws.onclose = (event) => {
        console.log('🔌 WebSocket disconnected:', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean
        });
        this.connected.set(false);
        // Reconnect after 3 seconds
        console.log('⏳ Will reconnect in 3 seconds...');
        setTimeout(() => this.connect(url), 3000);
      };
    } catch (error) {
      console.error('❌ Failed to connect WebSocket:', error);
    }
  }
  
  handleMessage(message) {
    console.log('🔄 handleMessage:', message.type);
    this.data.update(current => {
      let updated;
      switch (message.type) {
        case 'initial_state':
          console.log('📦 Processing initial_state:', {
            brainSpaceClusters: message.brain_space?.clusters?.length || 0,
            controlsKeys: Object.keys(message.controls || {})
          });
          updated = {
            ...current,
            brainSpace: message.brain_space || current.brainSpace,
            controls: message.controls || current.controls
          };
          break;
        
        case 'activation_update':
          console.log('⚡ Processing activation_update:', {
            waveformLength: message.waveform?.length || 0,
            sparksCount: message.sparks?.length || 0,
            morphism: message.morphism
          });
          updated = {
            ...current,
            activation: {
              waveform: message.waveform || current.activation.waveform,
              sparks: message.sparks || current.activation.sparks,
              morphism: message.morphism || current.activation.morphism
            }
          };
          break;
        
        case 'brain_space_update':
          console.log('🧠 Processing brain_space_update:', {
            clustersCount: message.brain_space?.clusters?.length || 0
          });
          updated = {
            ...current,
            brainSpace: message.brain_space || current.brainSpace
          };
          break;
        
        case 'event':
          console.log('📝 Processing event:', message.event);
          updated = {
            ...current,
            events: [...current.events.slice(-99), message.event]
          };
          break;
        
        default:
          console.log('⚠️  Unknown message type:', message.type);
          updated = current;
      }
      return updated;
    });
  }
  
  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('📤 Sending WebSocket message:', message);
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('⚠️  Cannot send message, WebSocket not open:', {
        ws: !!this.ws,
        readyState: this.ws?.readyState
      });
    }
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Singleton instance
let websocketStoreInstance = null;

export function getWebSocketStore() {
  if (!websocketStoreInstance) {
    websocketStoreInstance = new WebSocketStore();
  }
  return websocketStoreInstance;
}

// Export the singleton instance
export const websocketStore = getWebSocketStore();

// Export the stores directly for easier use in components
export const dataStore = websocketStore.data;
export const connectedStore = websocketStore.connected;

// Derived stores for convenience
export const clustersStore = derived(dataStore, $data => $data?.brainSpace?.clusters || []);
export const memoriesStore = derived(dataStore, $data => $data?.brainSpace?.memories || []);
export const activationStore = derived(dataStore, $data => $data?.activation || { waveform: [], sparks: [], morphism: 'gamma' });

