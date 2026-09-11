"""Strict workbench contracts. Drafts retain partial observations."""
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
RecordID = Annotated[str, Field(pattern=r'^(team|player|possession|event|frame)-[0-9a-f]{32}$')]
Digest = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
SourceID = Annotated[str, Field(pattern=r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$')]
Notes = Annotated[str, Field(max_length=2000)]

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    @field_validator('schema_version', mode='before', check_fields=False)
    @classmethod
    def integer_version(cls,value):
        if type(value) is not int: raise ValueError('schema_version must be an integer')
        return value

class Team(Strict):
    name: Annotated[str, Field(max_length=80)] = ''
    color: Annotated[str, Field(max_length=40)] = ''
    evidence_seconds: Seconds | None = None

class Player(Strict):
    team_id: RecordID | None = None
    jersey_number: Annotated[str, Field(pattern=r'^[0-9]{1,3}$')] | None = None
    number_status: Literal['not_reviewed','readable','unreadable','offscreen','unclear'] = 'not_reviewed'
    evidence_seconds: Seconds | None = None
    notes: Notes = ''
    @model_validator(mode='after')
    def number(self):
        if (self.number_status == 'readable') != (self.jersey_number is not None):
            raise ValueError('readable number requires digits; other states require null')
        return self

class Interval(Strict):
    start_seconds: Seconds | None = None
    end_seconds: Seconds | None = None
    @model_validator(mode='after')
    def ordered(self):
        if self.start_seconds is not None and self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError('end must follow start')
        return self

class Possession(Interval):
    team_id: RecordID | None = None
    attack_direction: Literal['left','right','unclear'] = 'unclear'
    notes: Notes = ''

class Event(Interval):
    event_type: Literal['shot','pass','rebound','turnover','foul','substitution','screen','defense','other','unclear'] = 'shot'
    possession_id: RecordID | None = None
    actor_id: RecordID | None = None
    receiver_id: RecordID | None = None
    passer_id: RecordID | None = None
    last_pass_seconds: Seconds | None = None
    incoming_player_id: RecordID | None = None
    outgoing_player_id: RecordID | None = None
    shot_type: Literal['two','three','free_throw','non_shot','unclear'] | None = None
    outcome: Literal['made','miss','not_applicable','unclear'] | None = None
    scoring_decision: Literal['counted','not_counted','not_applicable','unclear'] | None = None
    play_context: Literal['in_play','foul_on_shot','after_whistle','other_dead_ball','not_applicable','unclear'] | None = None
    presentation: Literal['live','replay','unclear'] | None = None
    boundary_status: Literal['complete','too_short','unclear'] | None = None
    defensive_action: Literal['on_ball','help','switch','trap','zone','man_to_man','unclear'] | None = None
    notes: Notes = ''
    @model_validator(mode='after')
    def dead_ball(self):
        if self.scoring_decision == 'counted' and self.play_context in ('after_whistle','other_dead_ball'):
            raise ValueError('new dead-ball shot cannot have counted points')
        return self

class Point(Strict):
    point_id: Annotated[str, Field(pattern=r'^[0-9a-f]{32}$')]
    entity: Literal['player','ball']
    player_id: RecordID | None = None
    x: Annotated[float, Field(ge=0,le=1)] | None = None
    y: Annotated[float, Field(ge=0,le=1)] | None = None
    visibility: Literal['visible','occluded','offscreen','unclear']
    anchor: Literal['floor_contact','image_center']
    notes: Notes = ''
    @model_validator(mode='after')
    def coordinates(self):
        if (self.x is None) != (self.y is None):
            raise ValueError('coordinates must be paired')
        if self.visibility == 'visible' and self.x is None:
            raise ValueError('visible requires coordinates')
        if self.visibility == 'offscreen' and self.x is not None:
            raise ValueError('offscreen requires null coordinates')
        if self.entity == 'ball' and (self.anchor != 'image_center' or self.player_id is not None):
            raise ValueError('ball requires image_center and no player identity')
        return self

class Frame(Strict):
    timestamp_seconds: Seconds | None = None
    event_id: RecordID | None = None
    possession_id: RecordID | None = None
    notes: Notes = ''
    points: Annotated[list[Point], Field(max_length=32)] = Field(default_factory=list)
    @model_validator(mode='after')
    def unique_points(self):
        ids = [p.point_id for p in self.points]
        players = [p.player_id for p in self.points if p.player_id]
        if len(ids)!=len(set(ids)) or len(players)!=len(set(players)):
            raise ValueError('duplicate point or known player')
        return self

class RequestBase(Strict):
    schema_version: Literal[1]
    workspace_revision: Digest
    source_id: SourceID
    record_id: RecordID
    expected_revision: Annotated[int, Field(ge=0, le=1000000)]
    status: Literal['draft','human_reviewed']
    archived: bool

class TeamRequest(RequestBase):
    kind: Literal['team']
    data: Team
class PlayerRequest(RequestBase):
    kind: Literal['player']
    data: Player
class PossessionRequest(RequestBase):
    kind: Literal['possession']
    data: Possession
class EventRequest(RequestBase):
    kind: Literal['event']
    data: Event
class FrameRequest(RequestBase):
    kind: Literal['frame']
    data: Frame

RecordRequest = Annotated[Union[TeamRequest,PlayerRequest,PossessionRequest,EventRequest,FrameRequest], Field(discriminator='kind')]

class ProgressRequest(Strict):
    schema_version: Literal[1]
    workspace_revision: Digest
    source_id: SourceID
    position_seconds: Seconds
    selected_record_id: RecordID | None = None
    playback_rate: Annotated[float, Field(ge=.25, le=4)] = 1.

DATA_MODELS = {'team':Team,'player':Player,'possession':Possession,'event':Event,'frame':Frame}

def reviewed_complete(kind, data, *, legacy_version=None):
    if kind=='team': return bool(data['name'].strip())
    if kind=='player': return data['number_status']!='not_reviewed'
    if kind=='frame': return data['timestamp_seconds'] is not None and bool(data['points'])
    if data['start_seconds'] is None or data['end_seconds'] is None: return False
    if kind=='possession': return bool(data['team_id'] or data['notes'].strip())
    required=['presentation','boundary_status']
    if data['event_type']=='shot':
        required+=['shot_type','outcome']
        if legacy_version != 1: required+=['scoring_decision','play_context']
    return all(data.get(k) is not None for k in required) and (data['event_type'] not in ('other','unclear') or bool(data['notes'].strip()))


class PublicSource(Strict):
    source_id: SourceID
    title: Annotated[str, Field(min_length=1, max_length=200)]
    sha256: Digest
    duration_seconds: Annotated[float, Field(gt=0)]
    byte_size: Annotated[int, Field(gt=0)]
    review_allowed: bool
    role: Annotated[str, Field(min_length=1,max_length=100)]
    source_group: Annotated[str, Field(min_length=1,max_length=200)]
    training_rights: Literal['unknown']
    media_url: str | None
    fps: Annotated[float, Field(gt=0,le=1000)] | None = None
    @model_validator(mode='after')
    def role_binding(self):
        if self.review_allowed and self.role!='development_review': raise ValueError('source role is not authorized')
        if self.media_url != (f'/media/{self.source_id}' if self.review_allowed else None): raise ValueError('invalid media URL')
        return self

class PublicWorkspace(Strict):
    schema_version: Literal[1]
    workspace_id: SourceID
    title: Annotated[str, Field(min_length=1,max_length=200)]
    sources: Annotated[list[PublicSource], Field(min_length=1,max_length=100)]

class LegacyContext(Strict):
    batch_revision: Digest
    example_id: SourceID
    event_id: Annotated[str, Field(pattern=r'^(primary|event-[0-9a-f]{32})$')]
    source_id: SourceID
    source_sha256: Digest
    source_start_seconds: Seconds
    source_end_seconds: Seconds

class LegacyMetadata(LegacyContext):
    schema_version: Literal[1,2,3]
    label_status: Literal['draft','human_reviewed']

class StoredRecord(Strict):
    record_id: RecordID
    revision: Annotated[int, Field(ge=1,le=1000001)]
    source_id: SourceID
    source_sha256: Digest
    kind: Literal['team','player','possession','event','frame']
    status: Literal['draft','human_reviewed']
    archived: bool
    data: dict
    created_at: Annotated[str, Field(max_length=40)]
    updated_at: Annotated[str, Field(max_length=40)]
    origin: Literal['human','legacy']
    needs_review: bool
    legacy: LegacyMetadata | None = None

class StoredTeam(StoredRecord):
    kind: Literal['team']
    data: Team
class StoredPlayer(StoredRecord):
    kind: Literal['player']
    data: Player
class StoredPossession(StoredRecord):
    kind: Literal['possession']
    data: Possession
class StoredEvent(StoredRecord):
    kind: Literal['event']
    data: Event
class StoredFrame(StoredRecord):
    kind: Literal['frame']
    data: Frame
StoredRecordUnion = Annotated[Union[StoredTeam,StoredPlayer,StoredPossession,StoredEvent,StoredFrame], Field(discriminator='kind')]

class BackupHeader(Strict):
    type: Literal['header']
    format: Literal['videoscope-workbench']
    schema_version: Literal[1]
    workspace_revision: Digest
    workspace: PublicWorkspace
class BackupRecord(Strict):
    type: Literal['record']
    record: StoredRecordUnion
class BackupLegacy(Strict):
    type: Literal['legacy']
    legacy_id: Annotated[str, Field(pattern=r'^event-[0-9a-f]{32}:[1-9][0-9]{0,6}$')]
    context: LegacyContext
    sha256: Digest
    raw_base64: Annotated[str, Field(max_length=21848)]
class StoredProgress(ProgressRequest):
    updated_at: Annotated[str, Field(max_length=40)]
class BackupProgress(Strict):
    type: Literal['progress']
    progress: StoredProgress
class BackupCounts(Strict):
    records: Annotated[int, Field(ge=0)]
    legacy: Annotated[int, Field(ge=0)]
    progress: Annotated[int, Field(ge=0)]
class BackupTrailer(Strict):
    type: Literal['trailer']
    counts: BackupCounts
    sha256: Digest
BackupLine = Annotated[Union[BackupHeader,BackupRecord,BackupLegacy,BackupProgress,BackupTrailer], Field(discriminator='type')]
