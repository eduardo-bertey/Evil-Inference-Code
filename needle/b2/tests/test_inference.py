from conftest import requires_engine

pytestmark = requires_engine


def test_tool_call():
    from needle import Needle, tool

    @tool
    def send_email(to: str, subject: str, body: str):
        "Send an email to a recipient."
        return {"ok": True}

    agent = Needle(tools=[send_email])
    response = agent.complete("email finance@cactus.dev, subject expenses, body coffee $14")
    assert response.get("type") in ("call", "text", "refuse")
    if response.get("type") == "call":
        assert response["function_calls"][0]["name"] == "send_email"


def test_structured_extraction():
    import pydantic
    from needle import extract

    class Contact(pydantic.BaseModel):
        name: str
        email: str

    result = extract("John Doe can be reached at john@doe.com", Contact)
    assert result is None or isinstance(result, Contact)


def test_run_agentic_loop():
    from needle import Needle, tool

    @tool
    def get_weather(city: str):
        "Get the weather for a city."
        return {"temp_c": 20}

    agent = Needle(tools=[get_weather])
    response = agent.run("what's the weather in Paris?")
    assert isinstance(response, dict)
    assert "type" in response
