import os
import base64
import anthropic
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an expert fashion designer and pattern maker with decades of experience creating sewing patterns.

When given a photo of a garment, analyze it carefully and provide:

1. **Garment Identification** - What type of garment it is (dress, blouse, skirt, pants, jacket, etc.)
2. **Style Analysis** - Key design features (neckline, sleeves, silhouette, closures, pockets, etc.)
3. **Fabric Recommendations** - Best fabric types for recreating this garment
4. **Sewing Pattern Instructions** - A detailed, usable sewing pattern including:
   - List of pattern pieces needed
   - Measurements and sizing guidance (provide for size S/M/L or standard measurements)
   - Step-by-step construction instructions
   - Seam allowance recommendations
   - Special techniques or notions required (zippers, buttons, interfacing, etc.)
5. **Difficulty Level** - Beginner / Intermediate / Advanced with reasoning
6. **Estimated Yardage** - Fabric requirements for sizes S, M, L

Format your response clearly with sections and subsections. Be specific and practical — this should be something a home sewer can actually follow."""


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image_data = data["image"]
    # Strip data URL prefix if present (e.g., "data:image/jpeg;base64,...")
    if "," in image_data:
        media_type_part, image_data = image_data.split(",", 1)
        # Extract media type from data URL
        media_type = media_type_part.split(":")[1].split(";")[0] if ":" in media_type_part else "image/jpeg"
    else:
        media_type = data.get("media_type", "image/jpeg")

    try:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Please analyze this garment and provide a detailed sewing pattern so I can recreate it.",
                        },
                    ],
                }
            ],
        ) as stream:
            full_response = stream.get_final_message()

        result_text = ""
        for block in full_response.content:
            if block.type == "text":
                result_text = block.text
                break

        return jsonify({"pattern": result_text})

    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid API key. Please set ANTHROPIC_API_KEY environment variable."}), 401
    except anthropic.BadRequestError as e:
        return jsonify({"error": f"Could not process the image: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
