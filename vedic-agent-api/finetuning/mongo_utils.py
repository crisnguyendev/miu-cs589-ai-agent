from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

class MongoDBConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
            # Load environment variables
            load_dotenv()
            mongo_uri = os.getenv("MONGODB_URI")
            if not mongo_uri:
                raise ValueError("MONGODB_URI not set in environment variables")
            # Initialize MongoClient with reconnection settings
            cls._instance._client = MongoClient(
                mongo_uri,
                tls=True,
                tlsAllowInvalidCertificates=False,
                serverSelectionTimeoutMS=30000,  # 30 seconds timeout
                retryWrites=True,
                retryReads=True
            )
            cls._instance._db = cls._instance._client["vedic-agent"]
            cls._instance._feedback_collection = cls._instance._db["feedback"]
            logger.info("MongoDB connection initialized")
        return cls._instance

    @property
    def client(self):
        # Check if client is closed and reinitialize if necessary
        if self._client is None or not self._client.is_primary:
            logger.warning("MongoClient is closed or disconnected. Reconnecting...")
            load_dotenv()
            mongo_uri = os.getenv("MONGODB_URI")
            self._client = MongoClient(
                mongo_uri,
                tls=True,
                tlsAllowInvalidCertificates=False,
                serverSelectionTimeoutMS=30000,
                retryWrites=True,
                retryReads=True
            )
            self._db = self._client["vedic-agent"]
            self._feedback_collection = self._db["feedback"]
            logger.info("MongoDB connection reestablished")
        return self._client

    @property
    def feedback_collection(self):
        # Reinitialize the connection if feedback_collection is None
        if self._feedback_collection is None:
            logger.warning("Feedback collection is None. Reinitializing MongoDB connection...")
            load_dotenv()
            mongo_uri = os.getenv("MONGODB_URI")
            if not mongo_uri:
                raise ValueError("MONGODB_URI not set in environment variables")
            self._client = MongoClient(
                mongo_uri,
                tls=True,
                tlsAllowInvalidCertificates=False,
                serverSelectionTimeoutMS=30000,
                retryWrites=True,
                retryReads=True
            )
            self._db = self._client["vedic-agent"]
            self._feedback_collection = self._db["feedback"]
            logger.info("MongoDB connection reestablished for feedback collection")
        return self._feedback_collection

    def close(self):
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")
            self._client = None
            self._db = None
            self._feedback_collection = None

# Singleton instance
mongo_connection = MongoDBConnection()