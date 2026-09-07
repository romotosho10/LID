"""Persistence layer — versioned fingerprint storage in Postgres.

Replaces the local-JSON-file approach in the original api.py. Every time
a fingerprint is (re)mined for a table, a new row is inserted rather than
overwriting the old one — this is what makes "when did this invariant
first start drifting" an answerable question instead of losing history
on every remine.

Schema:
    tenants(id, name, api_key)
    fingerprints(id, tenant_id, table_name, version, invariants JSON, created_at)
    violations(id, fingerprint_id, batch_label, invariant_type, target,
               old_support, new_support, re_derived_formula, explanation, created_at)
"""
from __future__ import annotations
import datetime as dt
import json
import os

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

Base = declarative_base()


class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    api_key = Column(String, unique=True, nullable=False, index=True)

    fingerprints = relationship("Fingerprint", back_populates="tenant")


class Fingerprint(Base):
    __tablename__ = "fingerprints"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    table_name = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False)
    invariants = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    tenant = relationship("Tenant", back_populates="fingerprints")
    violations = relationship("Violation", back_populates="fingerprint")


class Violation(Base):
    __tablename__ = "violations"
    id = Column(Integer, primary_key=True)
    fingerprint_id = Column(Integer, ForeignKey("fingerprints.id"), nullable=False)
    batch_label = Column(String, nullable=True)
    invariant_type = Column(String, nullable=False)
    target = Column(String, nullable=True)
    old_support = Column(String, nullable=True)
    new_support = Column(String, nullable=True)
    re_derived_formula = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    fingerprint = relationship("Fingerprint", back_populates="violations")


def get_engine(database_url: str | None = None):
    url = database_url or os.environ.get("DATABASE_URL", "sqlite:///lid_local.db")
    return create_engine(url, future=True)


def init_db(engine):
    Base.metadata.create_all(engine)


def get_session(engine) -> Session:
    return sessionmaker(bind=engine, future=True)()


# --- tenant helpers -----------------------------------------------------

def get_or_create_tenant(session: Session, name: str, api_key: str) -> Tenant:
    tenant = session.query(Tenant).filter_by(api_key=api_key).one_or_none()
    if tenant is None:
        tenant = Tenant(name=name, api_key=api_key)
        session.add(tenant)
        session.commit()
    return tenant


def get_tenant_by_key(session: Session, api_key: str) -> Tenant | None:
    return session.query(Tenant).filter_by(api_key=api_key).one_or_none()


# --- fingerprint helpers -------------------------------------------------

def save_fingerprint_version(session: Session, tenant: Tenant, table_name: str, invariants) -> Fingerprint:
    latest = get_latest_fingerprint(session, tenant, table_name)
    next_version = (latest.version + 1) if latest else 1
    fp = Fingerprint(
        tenant_id=tenant.id,
        table_name=table_name,
        version=next_version,
        invariants=invariants,
    )
    session.add(fp)
    session.commit()
    return fp


def get_latest_fingerprint(session: Session, tenant: Tenant, table_name: str) -> Fingerprint | None:
    return (
        session.query(Fingerprint)
        .filter_by(tenant_id=tenant.id, table_name=table_name)
        .order_by(Fingerprint.version.desc())
        .first()
    )


def get_fingerprint_history(session: Session, tenant: Tenant, table_name: str):
    return (
        session.query(Fingerprint)
        .filter_by(tenant_id=tenant.id, table_name=table_name)
        .order_by(Fingerprint.version.asc())
        .all()
    )


def record_violations(session: Session, fingerprint: Fingerprint, batch_label: str, explained_violations: list):
    rows = []
    for item in explained_violations:
        inv = item["invariant"]
        rows.append(Violation(
            fingerprint_id=fingerprint.id,
            batch_label=batch_label,
            invariant_type=inv.get("type"),
            target=inv.get("target"),
            old_support=str(inv.get("old_support", "")),
            new_support=str(inv.get("new_support", "")),
            re_derived_formula=item.get("re_derived", {}).get("formula") if item.get("re_derived") else None,
            explanation=item.get("explanation"),
        ))
    session.add_all(rows)
    session.commit()
    return rows
