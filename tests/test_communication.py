from HolonicTrader.holon_core import Message, Holon, Disposition
from HolonicTrader.agent_trader import TraderHolon
import time

# Mock Agents
class MockAgent(Holon):
    def receive_message(self, sender, content):
        if isinstance(content, Message):
            print(f"[{self.name}] Received {content.type}: {content.payload}")
            return f"ACK_{content.type}"

def test_communication():
    print("Testing Holonic Communication...")
    
    # 1. Setup
    observer = MockAgent("Observer", Disposition(0.5, 0.5))
    trader = TraderHolon("TraderNexus", sub_holons={'observer': observer})
    
    # 2. Test Message Creation
    msg = Message(sender=trader.name, type="FETCH_DATA", payload={"symbol": "BTC/USDT"})
    print(f"✅ Message Created: {msg}")
    
    # 3. Test Routing (Simulated)
    # In real world, Trader.run_cycle would trigger this.
    # Here we manually verify the protocol.
    
    # Sending TO sub-agent
    response = observer.receive_message(trader, msg)
    if response == "ACK_FETCH_DATA":
        print("✅ Message Delivery Confirmed")
    else:
        print("❌ Message Delivery Failed")
        
    # 4. Test Trader 'Adaptation' Logic
    print("Testing Trader Regime Adaptation...")
    trader._adapt_to_regime('CHAOTIC')
    print(f"Mode CHAOTIC -> Autonomy: {trader.disposition.autonomy}, Integration: {trader.disposition.integration}")
    
    if trader.disposition.integration == 0.9:
        print("✅ Trader adapted to CHAOS (High Integration)")
    else:
        print("❌ Trader failed to adapt to CHAOS")

if __name__ == "__main__":
    test_communication()
