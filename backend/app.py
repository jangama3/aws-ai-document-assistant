import json
import os
import boto3

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime")

KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]
MODEL_ARN = os.environ["MODEL_ARN"]


def lambda_handler(event, context):
    try:
        body = parse_body(event)
        question = body.get("question", "").strip()

        if not question:
            return build_response(400, {"error": "Missing required field: question"})

        result = bedrock_agent_runtime.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN,
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": 5
                        }
                    }
                }
            }
        )

        answer = result.get("output", {}).get("text", "")
        citations = extract_citations(result)

        return build_response(200, {
            "question": question,
            "answer": answer,
            "citations": citations
        })

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return build_response(500, {"error": str(e)})


def parse_body(event):
    if "body" in event:
        if isinstance(event["body"], str):
            return json.loads(event["body"])
        return event["body"]

    return event


def extract_citations(result):
    citations = []

    for citation in result.get("citations", []):
        for reference in citation.get("retrievedReferences", []):
            content = reference.get("content", {})
            location = reference.get("location", {})

            s3_uri = (
                location
                .get("s3Location", {})
                .get("uri")
            )

            citations.append({
                "source": s3_uri,
                "text": content.get("text", "")[:500]
            })

    return citations


def build_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": json.dumps(body)
    }