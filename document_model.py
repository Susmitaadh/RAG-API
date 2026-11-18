
from sqlalchemy import Column, Integer, String
from database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    saved_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    total_chunks = Column(Integer, nullable=False)


class InterviewBooking(Base):
    __tablename__ = "interview_bookings"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    interview_date = Column(String, nullable=False)
    interview_time = Column(String, nullable=False)
    status = Column(String, default="confirmed")
