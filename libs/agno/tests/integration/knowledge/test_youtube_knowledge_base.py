from unittest.mock import patch

import pytest

from agno.agent import Agent
from agno.document import Document
from agno.knowledge.youtube import YouTubeKnowledgeBase
from agno.vectordb.lancedb.lance_db import LanceDb


# Mock YouTube reader responses
def mock_read(video_url):
    if "video1" in video_url:
        return [
            Document(
                name="Video 1 - Part 1",
                content="This is a video about machine learning basics. We discuss neural networks and deep learning.",
                meta_data={"video_id": "video1", "segment": 1, "source": "YouTube"},
            ),
            Document(
                name="Video 1 - Part 2",
                content="In this segment we talk about training neural networks and backpropagation.",
                meta_data={"video_id": "video1", "segment": 2, "source": "YouTube"},
            ),
        ]
    elif "video2" in video_url:
        return [
            Document(
                name="Video 2 - Part 1",
                content="This tutorial explains how to implement convolutional neural networks (CNNs) for image classification.",
                meta_data={"video_id": "video2", "segment": 1, "source": "YouTube"},
            ),
            Document(
                name="Video 2 - Part 2",
                content="We demonstrate how to train the CNN on the MNIST dataset and evaluate its performance.",
                meta_data={"video_id": "video2", "segment": 2, "source": "YouTube"},
            ),
            Document(
                name="Video 2 - Part 3",
                content="Finally, we discuss transfer learning and how to use pre-trained models.",
                meta_data={"video_id": "video2", "segment": 3, "source": "YouTube"},
            ),
        ]
    return []


async def mock_async_read(video_url):
    # Reuse the synchronous mock implementation
    return mock_read(video_url)


@pytest.fixture
def mock_youtube_reader():
    with patch("agno.document.reader.youtube_reader.YouTubeReader.read", side_effect=mock_read), patch(
        "agno.document.reader.youtube_reader.YouTubeReader.async_read", side_effect=mock_async_read
    ):
        yield


def test_youtube_knowledge_base(mock_youtube_reader):
    vector_db = LanceDb(
        table_name="youtube_videos",
        uri="tmp/lancedb",
    )

    # Create a knowledge base with mock YouTube URLs
    knowledge_base = YouTubeKnowledgeBase(
        urls=["https://www.youtube.com/watch?v=video1", "https://www.youtube.com/watch?v=video2"],
        vector_db=vector_db,
    )

    knowledge_base.load(recreate=True)

    assert vector_db.exists()

    # We have 2 videos with 2 and 3 segments respectively
    expected_docs = 5
    assert vector_db.get_count() == expected_docs

    # Create and use the agent
    agent = Agent(knowledge=knowledge_base)
    response = agent.run("Tell me about neural networks", markdown=True)

    tool_calls = []
    for msg in response.messages:
        if msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    for call in tool_calls:
        if call.get("type", "") == "function":
            assert call["function"]["name"] == "search_knowledge_base"

    # Clean up
    vector_db.drop()


def test_youtube_knowledge_base_single_video(mock_youtube_reader):
    vector_db = LanceDb(
        table_name="youtube_single",
        uri="tmp/lancedb",
    )

    # Create a knowledge base with a single YouTube URL
    knowledge_base = YouTubeKnowledgeBase(
        urls=["https://www.youtube.com/watch?v=video1"],
        vector_db=vector_db,
    )

    knowledge_base.load(recreate=True)

    assert vector_db.exists()

    # The first video has 2 segments
    expected_docs = 2
    assert vector_db.get_count() == expected_docs

    # Clean up
    vector_db.drop()


@pytest.mark.asyncio
async def test_youtube_knowledge_base_async(mock_youtube_reader):
    vector_db = LanceDb(
        table_name="youtube_async",
        uri="tmp/lancedb",
    )

    # Create knowledge base
    knowledge_base = YouTubeKnowledgeBase(
        urls=["https://www.youtube.com/watch?v=video1", "https://www.youtube.com/watch?v=video2"],
        vector_db=vector_db,
    )

    await knowledge_base.aload(recreate=True)

    assert await vector_db.async_exists()

    # We have 2 videos with 2 and 3 segments respectively
    expected_docs = 5
    assert await vector_db.async_get_count() == expected_docs

    # Create and use the agent
    agent = Agent(knowledge=knowledge_base)
    response = await agent.arun("How do I implement CNNs for image classification?", markdown=True)

    tool_calls = []
    for msg in response.messages:
        if msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    for call in tool_calls:
        if call.get("type", "") == "function":
            assert call["function"]["name"] == "search_knowledge_base"

    # Check for CNN-related content in the response
    assert any(term in response.content.lower() for term in ["cnn", "convolutional", "classification"])

    # Clean up
    await vector_db.async_drop()


@pytest.mark.asyncio
async def test_youtube_knowledge_base_async_single_video(mock_youtube_reader):
    vector_db = LanceDb(
        table_name="youtube_async_single",
        uri="tmp/lancedb",
    )

    # Create knowledge base with a single YouTube URL
    knowledge_base = YouTubeKnowledgeBase(
        urls=["https://www.youtube.com/watch?v=video2"],
        vector_db=vector_db,
    )

    await knowledge_base.aload(recreate=True)

    assert await vector_db.async_exists()

    expected_docs = 3
    assert await vector_db.async_get_count() == expected_docs

    # Clean up
    await vector_db.async_drop()
