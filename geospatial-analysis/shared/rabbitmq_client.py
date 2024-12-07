import aiormq
import json


class RabbitMQClientInterface:
    queue_name: str = ""
    channel: aiormq.Channel = None

    def __init__(self, queue_name: str):
        self.queue_name = queue_name

    async def dispatch(self, message, routing_key: str = ""):
        pass

    async def finish(self, tag):
        pass

    def on_message(self, message: aiormq.abc.DeliveredMessage):
        """Load in the file for extracting text."""
        pass


class RabbitMQClientConnection:
    def __init__(self) -> None:
        self.connection = None
        self.channel = None
        self.client_connection = self

    async def setup(self, hostname: str) -> dict:
        self.connection = await aiormq.connect(url=hostname)
        self.channel = await self.connection.channel()

        await self.channel.basic_qos(prefetch_count=1)
        return {"connection": self.connection, "channel": self.channel}

    async def register(self, queue_exec: RabbitMQClientInterface):
        declare_ok = await self.channel.queue_declare(
            queue=queue_exec.queue_name,
            durable=True
        )

        queue_exec.queue_name = queue_exec.queue_name
        queue_exec.channel = self.channel
        await self.channel.basic_consume(declare_ok.queue, consumer_callback=queue_exec.on_message, no_ack=False)
        print(f'Registered consumer {declare_ok.queue}')


if __name__ == "__main__":
    pass
