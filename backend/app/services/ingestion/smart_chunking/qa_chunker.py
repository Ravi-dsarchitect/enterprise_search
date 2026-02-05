"""
QAChunker - Specialized chunker for Q&A format documents.

Keeps question-answer pairs together as atomic units.
"""

import re
from typing import List, Dict, Optional, Tuple

from app.services.ingestion.interfaces import (
    Chunker,
    ParsedDocument,
    ParsedBlock,
    StructuredChunk,
)
from app.services.ingestion.chunkers import split_into_sentences
from app.core.config import settings


class QAChunker(Chunker):
    """
    Specialized chunker for FAQ/Q&A format documents.

    Strategy:
    1. Detect Q&A pairs using patterns
    2. Keep each Q&A pair together as one chunk
    3. Group related Q&As under topic headings
    4. Split long answers while preserving question context
    """

    # Question patterns
    Q_PATTERNS = [
        r"^\s*(?:Q[\s.:]*\d*[:.\s]+)(.+\?)\s*$",              # Q: or Q1:
        r"^\s*(?:Question[\s.:]*\d*[:.\s]+)(.+\?)\s*$",       # Question:
        r"^\s*(\d+\.\s+.+\?)\s*$",                            # 1. Question?
        r"^\s*([-•]\s+.+\?)\s*$",                             # - Question?
    ]

    # Answer patterns
    A_PATTERNS = [
        r"^\s*(?:A[\s.:]*\d*[:.\s]+)",                        # A: or A1:
        r"^\s*(?:Answer[\s.:]*\d*[:.\s]+)",                   # Answer:
        r"^\s*(?:Ans[\s.:]*[:.\s]+)",                         # Ans:
    ]

    def __init__(
        self,
        chunk_size: int = None,
        max_qa_chunk: int = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.max_qa_chunk = max_qa_chunk or (self.chunk_size * 2)

    def chunk(self, text: str) -> List[str]:
        """Chunk plain text by detecting Q&A pairs."""
        pairs = self._extract_qa_pairs_from_text(text)

        if not pairs:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=100,
            )
            return splitter.split_text(text)

        return [f"Q: {q}\nA: {a}" for q, a, _, _ in pairs]

    def chunk_structured(
        self,
        doc: ParsedDocument,
        doc_metadata: Dict = None,
    ) -> List[StructuredChunk]:
        """Create structured chunks preserving Q&A pairs."""
        doc_metadata = doc_metadata or {}
        chunks = []

        pairs = self._extract_qa_pairs_from_blocks(doc.blocks)
        current_topic = None

        for question, answer, topic, page_number in pairs:
            if topic:
                current_topic = topic

            qa_text = f"Q: {question}\nA: {answer}"

            # Handle oversized Q&A pairs
            if len(qa_text) > self.max_qa_chunk:
                split_chunks = self._split_long_qa(
                    question, answer, current_topic, page_number
                )
                chunks.extend(split_chunks)
            else:
                chunks.append(StructuredChunk(
                    text=qa_text,
                    section_type="faq",
                    content_type="qa_pair",
                    heading=current_topic,
                    page_number=page_number,
                    parent_text=qa_text[:2000],
                    metadata={"question": question[:200], "topic": current_topic},
                ))

        # If no Q&A pairs found, fall back to structured chunking
        if not chunks:
            from app.services.ingestion.chunkers import StructuredChunker

            fallback = StructuredChunker(chunk_size=self.chunk_size)
            return fallback.chunk_structured(doc, doc_metadata)

        return chunks

    def _extract_qa_pairs_from_text(
        self,
        text: str,
    ) -> List[Tuple[str, str, Optional[str], int]]:
        """Extract Q&A pairs from plain text."""
        lines = text.split("\n")
        pairs = []

        current_question = None
        current_answer_lines = []
        current_topic = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this is a question
            is_question = False
            question_text = None

            for pattern in self.Q_PATTERNS:
                match = re.match(pattern, stripped, re.IGNORECASE)
                if match:
                    is_question = True
                    question_text = match.group(1).strip() if match.groups() else stripped
                    break

            # Also check for lines ending with ? that look like questions
            if not is_question and stripped.endswith("?") and len(stripped) > 10:
                # Avoid false positives by checking it's not part of an answer
                if not any(re.match(p, stripped, re.IGNORECASE) for p in self.A_PATTERNS):
                    is_question = True
                    question_text = stripped

            if is_question:
                # Save previous Q&A pair
                if current_question and current_answer_lines:
                    pairs.append((
                        current_question,
                        " ".join(current_answer_lines),
                        current_topic,
                        1,
                    ))

                current_question = question_text
                current_answer_lines = []
            else:
                # Check if this starts an answer
                is_answer_start = any(
                    re.match(p, stripped, re.IGNORECASE) for p in self.A_PATTERNS
                )

                if is_answer_start:
                    # Remove the answer marker
                    for pattern in self.A_PATTERNS:
                        stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE).strip()
                    if stripped:
                        current_answer_lines.append(stripped)
                elif current_question:
                    current_answer_lines.append(stripped)

        # Don't forget the last pair
        if current_question and current_answer_lines:
            pairs.append((
                current_question,
                " ".join(current_answer_lines),
                current_topic,
                1,
            ))

        return pairs

    def _extract_qa_pairs_from_blocks(
        self,
        blocks: List[ParsedBlock],
    ) -> List[Tuple[str, str, Optional[str], int]]:
        """Extract Q&A pairs from parsed blocks."""
        pairs = []
        current_topic = None
        current_question = None
        current_answer_lines = []
        current_page = 1

        for block in blocks:
            content = block.content.strip()
            if not content:
                continue

            # Check for topic heading
            if block.heading_level > 0:
                # Save previous pair
                if current_question and current_answer_lines:
                    pairs.append((
                        current_question,
                        " ".join(current_answer_lines),
                        current_topic,
                        current_page,
                    ))
                    current_question = None
                    current_answer_lines = []

                current_topic = content
                continue

            # Check if this is a question
            is_question = False
            question_text = content

            for pattern in self.Q_PATTERNS:
                match = re.match(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    is_question = True
                    question_text = match.group(1).strip() if match.groups() else content
                    break

            # Also check for content ending with ?
            if not is_question and content.strip().endswith("?"):
                is_question = True

            if is_question:
                # Save previous pair
                if current_question and current_answer_lines:
                    pairs.append((
                        current_question,
                        " ".join(current_answer_lines),
                        current_topic,
                        current_page,
                    ))

                current_question = question_text
                current_answer_lines = []
                current_page = block.page_number
            elif current_question:
                # This is answer content
                # Remove answer markers
                answer_text = content
                for pattern in self.A_PATTERNS:
                    answer_text = re.sub(pattern, "", answer_text, flags=re.IGNORECASE).strip()
                if answer_text:
                    current_answer_lines.append(answer_text)

        # Final pair
        if current_question and current_answer_lines:
            pairs.append((
                current_question,
                " ".join(current_answer_lines),
                current_topic,
                current_page,
            ))

        return pairs

    def _split_long_qa(
        self,
        question: str,
        answer: str,
        topic: Optional[str],
        page_number: int,
    ) -> List[StructuredChunk]:
        """Split a long Q&A pair while preserving question context."""
        chunks = []

        sentences = split_into_sentences(answer)
        current_part = []
        current_size = len(question) + 20  # Account for "Q: " and "A: "
        part_num = 1

        for sentence in sentences:
            if current_size + len(sentence) > self.chunk_size and current_part:
                answer_part = " ".join(current_part)
                suffix = f" (Part {part_num})" if part_num > 1 else ""
                chunk_text = f"Q: {question}\nA{suffix}: {answer_part}"

                chunks.append(StructuredChunk(
                    text=chunk_text,
                    section_type="faq",
                    content_type="qa_pair",
                    heading=topic,
                    page_number=page_number,
                    parent_text=f"Q: {question}\nA: {answer[:1500]}",
                    metadata={
                        "question": question[:200],
                        "topic": topic,
                        "part": part_num,
                    },
                ))

                current_part = []
                current_size = len(question) + 30
                part_num += 1

            current_part.append(sentence)
            current_size += len(sentence)

        # Final part
        if current_part:
            answer_part = " ".join(current_part)
            suffix = f" (Part {part_num})" if part_num > 1 else ""
            chunk_text = f"Q: {question}\nA{suffix}: {answer_part}"

            chunks.append(StructuredChunk(
                text=chunk_text,
                section_type="faq",
                content_type="qa_pair",
                heading=topic,
                page_number=page_number,
                parent_text=f"Q: {question}\nA: {answer[:1500]}",
                metadata={
                    "question": question[:200],
                    "topic": topic,
                    "part": part_num if part_num > 1 else None,
                },
            ))

        return chunks
