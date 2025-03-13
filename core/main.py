import asyncio
from core.orchestrator import Orchestrator

async def main():
    orchestrator = Orchestrator()
    await orchestrator.start()

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())