"""Legacy entry point. The Streamlit UI has been replaced by a static web
frontend served directly by FastAPI from frontend/web/.

Research prototype. NOT clinically validated.

Run the app with:
    uvicorn api.main:app --reload
and open http://127.0.0.1:8000 in a browser.
"""
import webbrowser


def main() -> None:
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
