import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from ...schemas import EventData, TrackingData


def _convert_datetime(kloppy_timestamp: timedelta, game_date: pd.Timestamp, period_id: int, verbose: bool = True):
    # Because kloppy timestamps are always relative to the start of the period, we add a timestamp offset
    # depending on the period_id to get a better timestamp approximation.
    timestamp_offset = (
        pd.Timedelta(minutes=45) if period_id == 2 else
        pd.Timedelta(minutes=90) if period_id == 3 else
        pd.Timedelta(minutes=105) if period_id == 4 else
        pd.Timedelta(0)
    )
    
    if kloppy_timestamp is None:
        return None
    if game_date is not None:
        return kloppy_timestamp + game_date + timestamp_offset
    else:
        if verbose:
            warnings.warn("Game date is None, using Unix epoch ('1975-01-01') as fall back date.")
        return kloppy_timestamp + pd.Timestamp('1975-01-01') + timestamp_offset

def players_from_kloppy(dataset: "EventDataset"):
    from kloppy.domain import Ground
    
    home_players, away_players = [], []
    for player in dataset.metadata.teams[0].players + dataset.metadata.teams[1].players:
        p = {
            "id": player.player_id,
            "full_name": player.name,
            "shirt_num": player.jersey_no,
            "position": (
                player
                .starting_position
                .position_group
                .value[0]
                .lower()
                .replace("attacker", "forward")
                .replace("unknown", "unspecified")
            ),
            "start_frame": -999,
            "end_frame": -999,
            "starter": player.starting,
        }
        if player.team.ground == Ground.HOME:
            home_players.append(p)
        else:
            away_players.append(p)
    return pd.DataFrame(home_players), pd.DataFrame(away_players)

def periods_from_kloppy(event_dataset, tracking_dataset) -> pd.DataFrame:
    if len(event_dataset.metadata.periods) != len(tracking_dataset.metadata.periods):
        min_periods = min(len(event_dataset.metadata.periods), len(tracking_dataset.metadata.periods))
        event_dataset.metadata.periods = event_dataset.metadata.periods[0:min_periods]
        tracking_dataset.metadata.periods = tracking_dataset.metadata.periods[0:min_periods]
        warnings.warn(f"Number of periods in event and tracking dataset do not match. Using the minimum ({min_periods}) periods from both datasets.")

    assert len(event_dataset.metadata.periods) == len(tracking_dataset.metadata.periods)

    game_date = tracking_dataset.metadata.date
    periods = []
    
    # the Game.periods object must always have 5 enties
    for i in range(1, 6):
        period_records_td = tracking_dataset.filter(lambda frame: frame.period.id == i)
        period_records_ed = event_dataset.filter(lambda frame: frame.period.id == i)

        if len(period_records_td.records) == 0:
            periods.append({
                "period_id": i,
                "start_frame": -999,
                "end_frame": -999,
                "start_timestamp_td": None,
                "end_timestamp_td": None,
                "start_timestamp_ed": None,
                "end_timestamp_ed": None,
            })
            continue
        
        period_td = tracking_dataset.metadata.periods[i - 1]
        period_ed = event_dataset.metadata.periods[i - 1]
        
        if isinstance(period_td.start_timestamp, timedelta) or period_td.start_timestamp is None:
            start_timestamp_td = _convert_datetime(period_records_td[0].timestamp, game_date, period_id=i, verbose=False)
        else:
            start_timestamp_td = period_td.start_timestamp

        if isinstance(period_td.end_timestamp, timedelta) or period_td.end_timestamp is None:
            end_timestamp_td = _convert_datetime(period_records_td[-1].timestamp, game_date, period_id=i, verbose=False)
        else:
            end_timestamp_td = period_td.end_timestamp

        if isinstance(period_ed.start_timestamp, timedelta) or period_ed.start_timestamp is None:
            start_timestamp_ed = _convert_datetime(period_records_ed[0].timestamp, game_date, period_id=i, verbose=False)
        else:
            start_timestamp_ed = period_ed.start_timestamp

        if isinstance(period_ed.end_timestamp, timedelta) or period_ed.end_timestamp is None:
            end_timestamp_ed = _convert_datetime(period_records_ed[-1].timestamp, game_date, period_id=i, verbose=False)
        else:
            end_timestamp_ed = period_ed.end_timestamp
    
        periods.append({
            "period_id": i,
            "start_frame": period_records_td[0].frame_id,
            "end_frame": period_records_td[-1].frame_id,
            "start_timestamp_td": start_timestamp_td,
            "end_timestamp_td": end_timestamp_td,
            "start_timestamp_ed": start_timestamp_ed,
            "end_timestamp_ed": end_timestamp_ed,
        })
    
    return pd.DataFrame(periods)

def convert_kloppy_tracking_dataset(tracking_dataset: "TrackingDataset") -> TrackingData:
    home_team, away_team = tracking_dataset.metadata.teams

    player_columns = {}
    for player in home_team.players + away_team.players:
        player_columns.update({f"{player.player_id}_x": f"{player.team.ground}_{player.jersey_no}_x"})
        player_columns.update({f"{player.player_id}_y": f"{player.team.ground}_{player.jersey_no}_y"})

    team_id_to_side = {
        home_team.team_id: "home",
        away_team.team_id: "away"
    }

    tracking_dataframe = (
        tracking_dataset
        .to_df(
            "frame_id",
            "period_id",
            "timestamp",
            "ball_state",
            "ball_owning_team_id",
            "ball_z",
            "*_x",
            "*_y",
            engine="pandas"
        )
        .assign(
            timestamp=lambda x: x.apply(
                lambda row: _convert_datetime(row["timestamp"], tracking_dataset.metadata.date, period_id=row["period_id"], verbose=False),
                axis=1
            ),
            team_possession=lambda x: x["ball_owning_team_id"].map(team_id_to_side),
            gametime_td=lambda x: x["timestamp"].dt.strftime("%M:%S")
        )
        .rename(columns={
            "frame_id": "frame",
            "ball_state": "ball_status",
            "timestamp": "datetime",
        } | player_columns)
        .drop(columns=["ball_owning_team_id"]) 
    )

    return TrackingData(
        tracking_dataframe,
        provider=tracking_dataset.metadata.provider.value,
        frame_rate=tracking_dataset.metadata.frame_rate,
    )

def convert_kloppy_event_dataset(event_dataset: "EventDataset") -> EventData:
    from kloppy.domain import (
        CarryResult,
        DuelResult,
        EventType,
        InterceptionResult,
        PassResult,
        ShotResult,
        TakeOnResult,
    )

    IS_SUCCESSFUL = [
        ShotResult.GOAL, 
        ShotResult.OWN_GOAL,
        PassResult.COMPLETE,
        TakeOnResult.COMPLETE,
        CarryResult.COMPLETE,
        DuelResult.WON,
        InterceptionResult.SUCCESS
    ]
    EVENT_MAP = {
        EventType.PASS.value: "pass",
        EventType.SHOT.value: "shot",
        EventType.CARRY.value: "dribble",
        EventType.TAKE_ON.value: "dribble"
    }

    home_team, away_team = event_dataset.metadata.teams
    players = home_team.players + away_team.players

    player_id_to_name = {player.player_id: player.name for player in players}

    event_data = (
        event_dataset
        .to_df(
            "period_id",
            "event_id",
            "timestamp",
            "player_id",
            "team_id",
            "coordinates_x",
            "coordinates_y",
            "event_type",
            "result",
            is_successful=lambda event: None if event.result is None else True if event.result in IS_SUCCESSFUL else False,
            minutes=lambda event: (int(event.timestamp.total_seconds()) % 3600 // 60) + (45 if event.period.id == 2 else 15 if event.period.id in [3, 4] else 0),
            seconds=lambda event: float(event.timestamp.total_seconds()) % 60,    
            engine="pandas"
        )
        .sort_values(by=['period_id','timestamp'], ascending=True)
        .reset_index(drop=True)
        .reset_index()
        .assign(
            timestamp=lambda x: x.apply(
                lambda row: _convert_datetime(row["timestamp"], event_dataset.metadata.date, period_id=row["period_id"], verbose=False),
                axis=1
            ),
            databallpy_event = lambda x: np.where(
                x['result'] == ShotResult.OWN_GOAL,
                'own_goal',
                x['event_type'].map(EVENT_MAP)
            ),
            player_name=lambda x: x["player_id"].map(player_id_to_name).astype(str),
            is_successful=lambda x: x['is_successful'].astype(pd.BooleanDtype()),

        )   
        .rename(columns={
            "frame_id": "frame",
            "ball_state": "ball_status",
            "ball_owning_team_id": "team_possession",
            "timestamp": "datetime",
            "coordinates_x": "start_x",
            "coordinates_y": "start_y",
            "event_id": "original_event_id",
            "index": "event_id",
            "event_type": "original_event"
        })
        .drop("result", axis=1)
    )

    return EventData(
        event_data, provider=event_dataset.metadata.provider.value
    )