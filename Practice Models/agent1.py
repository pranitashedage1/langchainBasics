import requests
from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert weather forecaster who always speaks in puns.
You have access to two tools:
- get_weather_for_location: use this to get real weather for a specific city
- get_user_location: use this if the user does not mention a city

Always respond in a punny, fun way. After getting weather data, give a complete
response with the actual temperature, conditions, humidity, and wind speed —
but make it funny and full of weather puns!"""

# ─── Weather Code Descriptions ─────────────────────────────────────────────────
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}

# ─── Tools ─────────────────────────────────────────────────────────────────────
@tool
def get_user_location() -> str:
    """Get the user's current location when they don't specify a city."""
    return "New York"

@tool
def get_weather_for_location(city: str) -> str:
    """Get real current weather for a given US city. Returns temperature, conditions, humidity, and wind speed."""
    try:
        # Step 1: Geocode city → lat/lon
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "country": "US"},
            timeout=10
        )
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return f"City '{city}' not found. Please check the spelling or try a nearby major city."

        result    = geo_data["results"][0]
        lat       = result["latitude"]
        lon       = result["longitude"]
        city_name = result["name"]
        state     = result.get("admin1", "")

        # Step 2: Fetch weather
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "America/New_York",
            },
            timeout=10
        )
        w = weather_resp.json().get("current", {})

        temp       = w.get("temperature_2m", "N/A")
        feels_like = w.get("apparent_temperature", "N/A")
        humidity   = w.get("relative_humidity_2m", "N/A")
        wind       = w.get("wind_speed_10m", "N/A")
        condition  = WEATHER_CODES.get(w.get("weather_code", -1), "Unknown")

        return (
            f"Weather in {city_name}, {state}:\n"
            f"  Condition : {condition}\n"
            f"  Temp      : {temp}°F (feels like {feels_like}°F)\n"
            f"  Humidity  : {humidity}%\n"
            f"  Wind      : {wind} mph"
        )

    except requests.RequestException as e:
        return f"Network error fetching weather: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

# ─── Tool Execution Helper ─────────────────────────────────────────────────────
TOOLS = {
    "get_weather_for_location": get_weather_for_location,
    "get_user_location": get_user_location,
}

def run_tool(tool_call):
    """Execute a tool call and return the result."""
    name   = tool_call["name"]
    inputs = tool_call.get("args") or tool_call.get("input") or {}
    fn     = TOOLS.get(name)
    if fn:
        return fn.invoke(inputs)
    return f"Unknown tool: {name}"

# ─── Agent Setup ───────────────────────────────────────────────────────────────
model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0.7,
).bind_tools(list(TOOLS.values()))

# ─── Interactive Chat Loop ─────────────────────────────────────────────────────
def chat():
    print("\n🌤️  Weather Forecaster (with Puns!) 🌤️")
    print("Ask me about the weather in any US city.")
    print("Type 'quit' or 'exit' to stop.\n")

    conversation = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Forecaster: Stay dry out there — or don't, living on the 'precipitation' edge! Goodbye! 👋")
            break

        conversation.append(HumanMessage(content=user_input))

        # Agentic loop: keep running until no more tool calls
        while True:
            response = model.invoke(conversation)
            conversation.append(response)

            # If there are tool calls, execute them
            if response.tool_calls:
                for tc in response.tool_calls:
                    print(f"  [🔧 Calling tool: {tc['name']} {tc.get('args', {})}]")
                    result = run_tool(tc)
                    from langchain_core.messages import ToolMessage
                    conversation.append(
                        ToolMessage(content=result, tool_call_id=tc["id"])
                    )
            else:
                # No more tool calls — we have the final answer
                print(f"\nForecaster: {response.content}\n")
                break

if __name__ == "__main__":
    chat()