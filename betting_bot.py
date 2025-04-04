import asyncio
import requests
import anthropic
import logging
import telegram
import nest_asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import re
import sys
from retry import retry
import pytz
import os
import random
import traceback
import json

# Configuration de base
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

@dataclass
class Config:
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str
    ODDS_API_KEY: str
    PERPLEXITY_API_KEY: str
    CLAUDE_API_KEY: str
    MAX_MATCHES: int = 5
    MIN_PREDICTIONS: int = 5

@dataclass
class Match:
    home_team: str
    away_team: str
    competition: str
    region: str
    commence_time: datetime
    priority: int = 0
    predicted_score1: str = ""
    predicted_score2: str = ""
    stats: dict = field(default_factory=dict)
    bookmaker_odds: Dict[str, float] = field(default_factory=dict)
    is_cup_match: bool = False  # Indique si c'est un match de coupe

@dataclass
class Prediction:
    region: str
    competition: str
    match: str
    time: str
    predicted_score1: str
    predicted_score2: str
    prediction: str
    confidence: int
    home_odds: float = 0.0
    draw_odds: float = 0.0
    away_odds: float = 0.0

class BettingBot:
    def __init__(self, config: Config):
        self.config = config
        self.bot = telegram.Bot(token=config.TELEGRAM_BOT_TOKEN)
        self.claude_client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        self.available_predictions = [
            "1X", "X2", "12", 
            "+1.5 buts", "+2.5 buts", "-3.5 buts",
            "1", "2",
            "-1.5 buts 1ère mi-temps", 
            "+0.5 but 1ère mi-temps", "+0.5 but 2ème mi-temps"
        ]
        
        # Définition des 5 grands championnats avec priorité maximale
        self.top5_leagues = {
            "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿": 1,  # Premier League
            "Championnat d'Espagne de Football 🇪🇸": 1,  # La Liga
            "Championnat d'Allemagne de Football 🇩🇪": 1,  # Bundesliga
            "Championnat d'Italie de Football 🇮🇹": 1,  # Serie A
            "Championnat de France de Football 🇫🇷": 1,  # Ligue 1
        }
        
        # Coupes des 5 grands championnats et coupes européennes - priorité 1 également
        self.top_cups = {
            # Coupes européennes majeures
            "Ligue des Champions de l'UEFA 🇪🇺": 1,
            "Ligue Europa de l'UEFA 🇪🇺": 1,
            
            # Coupes nationales majeures
            "FA Cup 🏴󠁧󠁢󠁥󠁮󠁧󠁿": 1,
            "EFL Cup 🏴󠁧󠁢󠁥󠁮󠁧󠁿": 1,
            "Copa del Rey 🇪🇸": 1,
            "Supercoupe d'Espagne 🇪🇸": 1,
            "DFB-Pokal 🇩🇪": 1,
            "Supercoupe d'Allemagne 🇩🇪": 1,
            "Coppa Italia 🇮🇹": 1,
            "Supercoupe d'Italie 🇮🇹": 1,
            "Coupe de France 🇫🇷": 1,
            "Coupe de la Ligue Française 🇫🇷": 1,
            "Trophée des Champions 🇫🇷": 1,
            
            # Autres coupes européennes
            "Ligue Conférence de l'UEFA 🇪🇺": 2,
            "Supercoupe de l'UEFA 🇪🇺": 1,
        }
        
        # Autres compétitions importantes
        self.other_leagues = {
            # Championnats secondaires
            "Championnat de Belgique de Football 🇧🇪": 2,
            "Championnat des Pays-Bas de Football 🇳🇱": 2,
            "Championnat du Portugal de Football 🇵🇹": 2,
            "Premier League Russe 🇷🇺": 2,
            "Super League Suisse 🇨🇭": 2,
            "Süper Lig Turque 🇹🇷": 2,
            "Championship Anglais 🏴󠁧󠁢󠁥󠁮󠁧󠁿": 2,
            "Ligue 2 Française 🇫🇷": 2,
            "Serie B Italienne 🇮🇹": 2,
            "Segunda División Espagnole 🇪🇸": 2,
            "2. Bundesliga Allemande 🇩🇪": 2,
            
            # Compétitions internationales
            "Coupe du Monde FIFA 🌍": 1,
            "Ligue des Nations UEFA 🇪🇺": 1,
            "Championnat d'Europe UEFA 🇪🇺": 1,
            "Copa America 🌎": 1,
            "Coupe d'Afrique des Nations 🌍": 1,
            
            # Autres championnats internationaux
            "MLS 🇺🇸": 3,
            "Liga MX 🇲🇽": 3,
            "J-League 🇯🇵": 3,
            "K-League 🇰🇷": 3,
            "A-League 🇦🇺": 3,
            "Chinese Super League 🇨🇳": 3,
            "Brasileirão 🇧🇷": 3,
            "Argentine Primera División 🇦🇷": 3
        }
        
        # Fusionner les dictionnaires pour avoir toutes les compétitions
        self.all_competitions = {**self.top5_leagues, **self.top_cups, **self.other_leagues}
        
        # Liste de mots-clés pour identifier les matchs de coupe
        self.cup_keywords = [
            "cup", "coupe", "copa", "pokal", "coppa", "trophy", "trophée", 
            "supercoupe", "supercup", "champions league", "europa", "conférence",
            "final", "finale", "semi-final", "quarter-final"
        ]
        
        print("Bot initialisé!")

    def _get_league_name(self, competition: str) -> str:
        """Normalise les noms de compétitions pour correspondre à notre nomenclature"""
        league_mappings = {
            # Grands championnats européens
            "Premier League": "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "EPL": "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "La Liga": "Championnat d'Espagne de Football 🇪🇸",
            "Primera Division": "Championnat d'Espagne de Football 🇪🇸",
            "Bundesliga": "Championnat d'Allemagne de Football 🇩🇪",
            "Serie A": "Championnat d'Italie de Football 🇮🇹",
            "Ligue 1": "Championnat de France de Football 🇫🇷",
            
            # Coupes européennes
            "Champions League": "Ligue des Champions de l'UEFA 🇪🇺",
            "UEFA Champions League": "Ligue des Champions de l'UEFA 🇪🇺",
            "Europa League": "Ligue Europa de l'UEFA 🇪🇺",
            "UEFA Europa League": "Ligue Europa de l'UEFA 🇪🇺",
            "Conference League": "Ligue Conférence de l'UEFA 🇪🇺",
            "UEFA Europa Conference League": "Ligue Conférence de l'UEFA 🇪🇺",
            "UEFA Super Cup": "Supercoupe de l'UEFA 🇪🇺",
            
            # Coupes nationales
            "FA Cup": "FA Cup 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "EFL Cup": "EFL Cup 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "Carabao Cup": "EFL Cup 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "Copa del Rey": "Copa del Rey 🇪🇸",
            "Supercopa de España": "Supercoupe d'Espagne 🇪🇸",
            "DFB-Pokal": "DFB-Pokal 🇩🇪",
            "DFL-Supercup": "Supercoupe d'Allemagne 🇩🇪",
            "Coppa Italia": "Coppa Italia 🇮🇹",
            "Supercoppa Italiana": "Supercoupe d'Italie 🇮🇹",
            "Coupe de France": "Coupe de France 🇫🇷",
            "Coupe de la Ligue": "Coupe de la Ligue Française 🇫🇷",
            "Trophée des Champions": "Trophée des Champions 🇫🇷",
            
            # Championnats européens secondaires
            "Belgian Pro League": "Championnat de Belgique de Football 🇧🇪",
            "Belgian First Division A": "Championnat de Belgique de Football 🇧🇪",
            "Eredivisie": "Championnat des Pays-Bas de Football 🇳🇱",
            "Primeira Liga": "Championnat du Portugal de Football 🇵🇹",
            "Russian Premier League": "Premier League Russe 🇷🇺",
            "Swiss Super League": "Super League Suisse 🇨🇭",
            "Turkish Super Lig": "Süper Lig Turque 🇹🇷",
            "Championship": "Championship Anglais 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "EFL Championship": "Championship Anglais 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "Ligue 2": "Ligue 2 Française 🇫🇷",
            "Serie B": "Serie B Italienne 🇮🇹",
            "Segunda Division": "Segunda División Espagnole 🇪🇸",
            "2. Bundesliga": "2. Bundesliga Allemande 🇩🇪",
            
            # Compétitions internationales
            "FIFA World Cup": "Coupe du Monde FIFA 🌍",
            "UEFA Nations League": "Ligue des Nations UEFA 🇪🇺",
            "UEFA European Championship": "Championnat d'Europe UEFA 🇪🇺",
            "UEFA Euro": "Championnat d'Europe UEFA 🇪🇺",
            "Africa Cup of Nations": "Coupe d'Afrique des Nations 🌍",
            "Copa America": "Copa America 🌎",
            
            # Autres championnats internationaux
            "Major League Soccer": "MLS 🇺🇸",
            "MLS": "MLS 🇺🇸",
            "Liga MX": "Liga MX 🇲🇽",
            "J League": "J-League 🇯🇵",
            "K League 1": "K-League 🇰🇷",
            "A-League": "A-League 🇦🇺",
            "Chinese Super League": "Chinese Super League 🇨🇳",
            "Brasileirão": "Brasileirão 🇧🇷",
            "Brazilian Serie A": "Brasileirão 🇧🇷",
            "Argentine Primera División": "Argentine Primera División 🇦🇷"
        }
        return league_mappings.get(competition, competition)

    def _is_cup_match(self, competition: str) -> bool:
        """Détermine si un match est une compétition de coupe"""
        # Vérifier si c'est une coupe connue directement
        if competition in self.top_cups:
            return True
        
        # Vérifier si le nom contient un mot-clé de coupe
        competition_lower = competition.lower()
        for keyword in self.cup_keywords:
            if keyword.lower() in competition_lower:
                return True
        
        return False

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def fetch_matches(self) -> List[Match]:
        """Récupère les matchs à venir en priorisant les 5 grands championnats et leurs coupes"""
        print("\n1️⃣ RÉCUPÉRATION DES MATCHS...")
        url = "https://api.the-odds-api.com/v4/sports/soccer/odds/"
        params = {
            "apiKey": self.config.ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            matches_data = response.json()
            print(f"✅ {len(matches_data)} matchs récupérés depuis l'API")

            current_time = datetime.now(timezone.utc)
            all_matches = []

            # Collecter tous les matchs des prochaines 48h
            for match_data in matches_data:
                try:
                    commence_time = datetime.fromisoformat(match_data["commence_time"].replace('Z', '+00:00'))
                    sport_title = match_data.get("sport_title", "Unknown")
                    competition = self._get_league_name(sport_title)
                    is_cup = self._is_cup_match(competition)
                    
                    # Récupérer les cotes des bookmakers
                    bookmaker_odds = {}
                    if "bookmakers" in match_data and len(match_data["bookmakers"]) > 0:
                        # Prendre le premier bookmaker disponible
                        bookmaker = match_data["bookmakers"][0]
                        for market in bookmaker.get("markets", []):
                            if market["key"] == "h2h":
                                for outcome in market["outcomes"]:
                                    if outcome["name"] == match_data["home_team"]:
                                        bookmaker_odds["home"] = outcome["price"]
                                    elif outcome["name"] == match_data["away_team"]:
                                        bookmaker_odds["away"] = outcome["price"]
                                    else:
                                        bookmaker_odds["draw"] = outcome["price"]
                    
                    # Filtrer sur les prochaines 48h
                    if 0 < (commence_time - current_time).total_seconds() <= 172800:  # 48 heures
                        # Déterminer la priorité selon la compétition et si c'est une coupe
                        if competition in self.top5_leagues:
                            priority = 1  # Top 5 championnats
                        elif is_cup and competition in self.top_cups:
                            priority = 1  # Coupes majeures
                        elif competition in self.all_competitions:
                            priority = self.all_competitions[competition]
                        else:
                            priority = 5  # Priorité la plus basse
                        
                        # Ajuster la priorité pour les matchs de coupe des 5 grands championnats
                        if is_cup and priority > 1:
                            # Vérifier si c'est une coupe d'un des 5 grands pays
                            country_keywords = ["angla", "españ", "españo", "espan", "franc", "italien", "itali", "german", "allemand", "deutsch"]
                            competition_lower = competition.lower()
                            for keyword in country_keywords:
                                if keyword in competition_lower:
                                    priority = max(1, priority - 1)  # Augmenter la priorité
                                    break
                        
                        match = Match(
                            home_team=match_data["home_team"],
                            away_team=match_data["away_team"],
                            competition=competition,
                            region=competition.split()[-1] if " " in competition else "⚽",
                            commence_time=commence_time,
                            priority=priority,
                            bookmaker_odds=bookmaker_odds,
                            is_cup_match=is_cup
                        )
                        all_matches.append(match)
                except Exception as e:
                    print(f"⚠️ Erreur lors du traitement d'un match: {str(e)}")
                    continue

            if not all_matches:
                print("❌ Aucun match trouvé pour les prochaines 48 heures")
                return []

            # Trier les matchs par priorité (les plus importantes d'abord)
            all_matches.sort(key=lambda x: (x.priority, x.commence_time))
            
            # Compte des matchs par priorité et type pour le log
            priority_counts = {}
            cup_counts = 0
            league_counts = 0
            
            for match in all_matches:
                priority_counts[match.priority] = priority_counts.get(match.priority, 0) + 1
                if match.is_cup_match:
                    cup_counts += 1
                else:
                    league_counts += 1
            
            print("\n📊 Répartition des matchs par priorité:")
            for priority, count in sorted(priority_counts.items()):
                priority_name = "Priorité Maximale" if priority == 1 else f"Priorité {priority}"
                print(f"  - {priority_name}: {count} matchs")
            
            print(f"  - Matchs de championnat: {league_counts}")
            print(f"  - Matchs de coupe: {cup_counts}")
            
            # Vérifier si nous avons suffisamment de matchs de priorité 1
            priority1_matches = [m for m in all_matches if m.priority == 1]
            print(f"\n✅ {len(priority1_matches)} matchs de priorité maximale trouvés")
            
            # Si nous avons suffisamment de matchs prioritaires, ne prendre que ceux-là
            if len(priority1_matches) >= self.config.MIN_PREDICTIONS:
                selected_matches = priority1_matches[:self.config.MAX_MATCHES]
                print(f"✅ Sélection de {len(selected_matches)} matchs prioritaires")
            else:
                # Sinon, compléter avec d'autres matchs de priorité supérieure
                selected_matches = priority1_matches
                remaining_needed = self.config.MIN_PREDICTIONS - len(selected_matches)
                
                # Ajouter des matchs de priorité 2 si nécessaire
                priority2_matches = [m for m in all_matches if m.priority == 2]
                if priority2_matches and len(selected_matches) < self.config.MIN_PREDICTIONS:
                    num_to_add = min(remaining_needed, len(priority2_matches))
                    selected_matches.extend(priority2_matches[:num_to_add])
                    remaining_needed -= num_to_add
                
                # Continuer avec priorité 3 si nécessaire
                if remaining_needed > 0:
                    priority3_matches = [m for m in all_matches if m.priority == 3]
                    if priority3_matches:
                        num_to_add = min(remaining_needed, len(priority3_matches))
                        selected_matches.extend(priority3_matches[:num_to_add])
                        remaining_needed -= num_to_add
                
                # Continuer avec d'autres priorités si toujours pas assez
                if remaining_needed > 0:
                    other_matches = [m for m in all_matches if m.priority > 3 and m not in selected_matches]
                    if other_matches:
                        num_to_add = min(remaining_needed, len(other_matches))
                        selected_matches.extend(other_matches[:num_to_add])
            
            # S'assurer de ne pas dépasser MAX_MATCHES
            selected_matches = selected_matches[:self.config.MAX_MATCHES]
            
            # Trier par heure de début
            selected_matches.sort(key=lambda x: x.commence_time)
            
            print(f"\n✅ {len(selected_matches)} matchs candidats sélectionnés:")
            for i, match in enumerate(selected_matches, 1):
                odds_info = ""
                if match.bookmaker_odds:
                    home_odds = match.bookmaker_odds.get("home", "N/A")
                    draw_odds = match.bookmaker_odds.get("draw", "N/A")
                    away_odds = match.bookmaker_odds.get("away", "N/A")
                    odds_info = f" [Cotes: {home_odds}-{draw_odds}-{away_odds}]"
                
                match_type = "🏆 COUPE" if match.is_cup_match else "🏅 CHAMP."
                print(f"  {i}. {match_type} {match.home_team} vs {match.away_team} ({match.competition}, Priorité: {match.priority}){odds_info}")
                
            return selected_matches

        except Exception as e:
            print(f"❌ Erreur lors de la récupération des matchs: {str(e)}")
            traceback.print_exc()
            return []

    @retry(tries=2, delay=3, backoff=2, logger=logger)
    def collect_match_data(self, match: Match) -> bool:
        """Collecte toutes les données brutes nécessaires pour l'analyse du match via Perplexity"""
        print(f"\n2️⃣ COLLECTE DE DONNÉES POUR {match.home_team} vs {match.away_team}")
        try:
            # Structure pour collecter les différents types de données
            match.stats = {
                "forme_recente": None,
                "confrontations_directes": None,
                "statistiques_detaillees": None,
                "absences_effectif": None,
                "contexte_match": None,
                "bookmaker_odds": None
            }
            
            # Ajout des cotes des bookmakers si disponibles
            if match.bookmaker_odds:
                odds_content = f"Cotes actuelles des bookmakers pour {match.home_team} vs {match.away_team}:\n"
                odds_content += f"- Victoire {match.home_team}: {match.bookmaker_odds.get('home', 'N/A')}\n"
                odds_content += f"- Match nul: {match.bookmaker_odds.get('draw', 'N/A')}\n"
                odds_content += f"- Victoire {match.away_team}: {match.bookmaker_odds.get('away', 'N/A')}"
                match.stats["bookmaker_odds"] = odds_content
                print("✅ Données de cotes des bookmakers ajoutées")
            
            # 1. Forme récente
            forme_prompt = f"""En tant que collecteur de données sportives factuel, fournir exclusivement ces statistiques précises et vérifiées pour le match {match.home_team} vs {match.away_team}:

1. FORME RÉCENTE DE {match.home_team}:
   - Résultats des 5 derniers matchs (tous formats confondus) avec date, compétition, adversaire et score exact
   - Forme actuelle sous format série (ex: VVNDV)
   - Moyenne de buts marqués et encaissés sur les 5 derniers matchs
   - Performance à domicile: pourcentage de victoires, défaites, nuls
   - Tendance offensive et défensive récente

2. FORME RÉCENTE DE {match.away_team}:
   - Résultats des 5 derniers matchs (tous formats confondus) avec date, compétition, adversaire et score exact
   - Forme actuelle sous format série (ex: VVNDV)
   - Moyenne de buts marqués et encaissés sur les 5 derniers matchs
   - Performance à l'extérieur: pourcentage de victoires, défaites, nuls
   - Tendance offensive et défensive récente

IMPORTANT: Format sous forme de liste avec données UNIQUEMENT factuelle, aucune analyse ou opinion."""

            forme_response = self._get_perplexity_response(forme_prompt)
            if forme_response:
                match.stats["forme_recente"] = forme_response
                print("✅ Données de forme récente collectées")
            else:
                print("❌ Échec de la collecte des données de forme récente")
                return False
            
            # 2. Confrontations directes
            h2h_prompt = f"""En tant que collecteur de données sportives factuel, fournir exclusivement les résultats des confrontations directes entre {match.home_team} et {match.away_team}:

1. HISTORIQUE DES CONFRONTATIONS:
   - Les 5 dernières rencontres directes avec date exacte, compétition, et score final
   - Bilan global: nombre de victoires pour chaque équipe et de matchs nuls
   - Nombre moyen de buts par match lors des confrontations directes
   - Nombre de matchs où les deux équipes ont marqué
   - Tendance historique: quelle équipe domine généralement?

IMPORTANT: Format sous forme de liste, uniquement les données brutes factuelles sans analyse personnelle."""

            h2h_response = self._get_perplexity_response(h2h_prompt)
            if h2h_response:
                match.stats["confrontations_directes"] = h2h_response
                print("✅ Données de confrontations directes collectées")
            else:
                print("⚠️ Pas de données de confrontations directes disponibles")
                
                # Pour les matchs de coupe, tenter d'obtenir des données spécifiques de coupe
                if match.is_cup_match:
                    cup_h2h_prompt = f"""En tant que collecteur de données sportives factuel, recherche spécifiquement les confrontations entre {match.home_team} et {match.away_team} en compétitions de coupe ou matchs à élimination directe:

1. HISTORIQUE DES CONFRONTATIONS EN COUPE:
   - Rencontres en coupe/matchs à enjeu direct avec dates et scores
   - Performance de chaque équipe dans ce type de compétition
   - Statistiques spécifiques aux matchs à élimination directe
   - Parcours des deux équipes dans la compétition actuelle ({match.competition})

Format: données factuelles uniquement."""

                    cup_h2h_response = self._get_perplexity_response(cup_h2h_prompt)
                    if cup_h2h_response:
                        match.stats["confrontations_directes"] = cup_h2h_response
                        print("✅ Données de confrontations en coupe collectées")
            
            # 3. Statistiques détaillées
            stats_prompt = f"""En tant que collecteur de données sportives factuel, fournir exclusivement ces statistiques précises et actuelles pour {match.home_team} et {match.away_team} dans la compétition {match.competition}:

1. STATISTIQUES DE BUTS:
   - Pourcentage exact de matchs avec plus de 1.5 buts pour chaque équipe cette saison
   - Pourcentage exact de matchs avec plus de 2.5 buts pour chaque équipe cette saison
   - Pourcentage exact de matchs avec moins de 3.5 buts pour chaque équipe cette saison
   - Pourcentage de matchs où les deux équipes ont marqué

2. STATISTIQUES DÉFENSIVES:
   - Pourcentage de clean sheets (matchs sans encaisser de but) pour chaque équipe
   - Nombre moyen de buts encaissés par match pour chaque équipe
   - Pourcentage de matchs où l'équipe a encaissé en première mi-temps

3. STATISTIQUES OFFENSIVES:
   - Nombre moyen de buts marqués par match pour chaque équipe
   - Pourcentage de matchs où l'équipe a marqué en première mi-temps
   - Répartition des buts par période (1ère/2ème mi-temps)

IMPORTANT: Fournir UNIQUEMENT des statistiques vérifiées et factuelles, aucune opinion ou analyse."""

            stats_response = self._get_perplexity_response(stats_prompt)
            if stats_response:
                match.stats["statistiques_detaillees"] = stats_response
                print("✅ Données statistiques détaillées collectées")
            else:
                print("❌ Échec de la collecte des statistiques détaillées")
                return False
            
            # 4. Absences et effectif
            absences_prompt = f"""En tant que collecteur de données sportives factuel, fournir exclusivement ces informations sur les effectifs pour le match {match.home_team} vs {match.away_team}:

1. ABSENCES CONFIRMÉES:
   - Liste des joueurs blessés ou suspendus pour {match.home_team}
   - Liste des joueurs blessés ou suspendus pour {match.away_team}
   - Date prévue de retour si connue

2. JOUEURS CLÉS:
   - Meilleurs buteurs de chaque équipe cette saison avec nombre de buts
   - Joueurs importants de retour de blessure récemment
   - Joueurs en forme exceptionnelle actuellement

IMPORTANT: Format liste, données factuelles uniquement, pas d'analyse d'impact."""

            absences_response = self._get_perplexity_response(absences_prompt)
            if absences_response:
                match.stats["absences_effectif"] = absences_response
                print("✅ Données sur les absences et effectifs collectées")
            else:
                print("⚠️ Pas de données d'absences disponibles")
            
            # 5. Contexte du match
            context_prompt = ""
            if match.is_cup_match:
                context_prompt = f"""En tant que collecteur de données sportives factuel, fournir exclusivement ces informations sur le contexte du match de coupe {match.home_team} vs {match.away_team} dans la compétition {match.competition}:

1. CONTEXTE DE LA COUPE:
   - Phase actuelle de la compétition (1/8, quart, demi-finale, etc.)
   - Règles spécifiques (match aller-retour, élimination directe, etc.)
   - Parcours des deux équipes jusqu'à présent dans cette compétition
   - Historique des équipes dans cette compétition (palmarès)

2. ENJEU DU MATCH:
   - Importance pour chaque équipe (qualification, trophée, etc.)
   - Motivation et priorité donnée à cette compétition
   - Matchs à venir dans d'autres compétitions (gestion possible)
   - Conditions particulières (stade, ambiance, météo)

IMPORTANT: Données factuelles uniquement, pas d'analyse ni de pronostic."""
            else:
                context_prompt = f"""En tant que collecteur de données sportives factuel, fournir exclusivement ces informations sur le contexte du match {match.home_team} vs {match.away_team} dans la compétition {match.competition}:

1. CONTEXTE SPORTIF:
   - Position actuelle au classement des deux équipes avec points exacts
   - Écart de points avec les positions clés (qualification européenne, relégation, etc.)
   - Enjeu spécifique du match pour chaque équipe
   - Matchs à venir dans le calendrier des équipes (fatigue potentielle)

2. CONTEXTE EXTERNE:
   - Stade où se déroule le match et affluence moyenne
   - Conditions météorologiques prévues s'il s'agit d'un match en extérieur
   - Historique récent de l'arbitre désigné si connu

IMPORTANT: Données factuelles uniquement, pas d'analyse ni de pronostic."""

            contexte_response = self._get_perplexity_response(context_prompt)
            if contexte_response:
                match.stats["contexte_match"] = contexte_response
                print("✅ Données sur le contexte du match collectées")
            else:
                print("⚠️ Pas de données de contexte disponibles")
            
            # Vérifier que nous avons au moins les données essentielles
            essential_data = ["forme_recente", "statistiques_detaillees"]
            missing_data = [data for data in essential_data if not match.stats.get(data)]
            
            if not missing_data:
                print("✅ Données suffisantes collectées pour l'analyse")
                return True
            else:
                print(f"❌ Données insuffisantes pour l'analyse. Manquant: {', '.join(missing_data)}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de la collecte des données: {str(e)}")
            traceback.print_exc()
            return False

    def _get_perplexity_response(self, prompt: str) -> Optional[str]:
        """Fonction utilitaire pour obtenir une réponse de Perplexity"""
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                    "temperature": 0.1  # Température basse pour des réponses factuelles
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ Erreur lors de l'appel à Perplexity: {str(e)}")
            return None

    @retry(tries=2, delay=3, backoff=2, logger=logger)
    def analyze_with_claude(self, match: Match) -> Optional[Prediction]:
        """Analyse complète du match avec Claude pour obtenir les scores probables et la prédiction optimale"""
        print(f"\n3️⃣ ANALYSE AVEC CLAUDE POUR {match.home_team} vs {match.away_team}")
        
        if not match.stats.get("forme_recente") or not match.stats.get("statistiques_detaillees"):
            print("❌ Données statistiques essentielles manquantes pour l'analyse")
            return None
        
        try:
            # Préparer les données pour Claude
            data_sections = []
            for section_name, content in match.stats.items():
                if content:
                    formatted_section = f"### {section_name.upper().replace('_', ' ')}\n{content}"
                    data_sections.append(formatted_section)
            
            data_content = "\n\n".join(data_sections)
            
            # ÉTAPE 1: Obtenir les scores probables avec un prompt très directif
            scores_prompt = f"""En tant qu'expert en prédiction de football, donne-moi EXACTEMENT deux scores probables pour le match {match.home_team} vs {match.away_team}.

IMPORTANT: Réponds UNIQUEMENT avec le format exact ci-dessous, sans aucune explication:

SCORE_1: X-Y
SCORE_2: Z-W

où X, Y, Z et W sont des nombres entiers (comme 1-0, 2-1, 1-1, etc.)."""

            scores_message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,  # Réduit pour limiter les explications
                temperature=0.1,
                messages=[{"role": "user", "content": scores_prompt}]
            )

            scores_response = scores_message.content[0].text.strip()
            
            # Extraire les deux scores avec regex plus robuste
            score1_match = re.search(r'SCORE_1:?\s*(\d+)[ -]+(\d+)', scores_response)
            score2_match = re.search(r'SCORE_2:?\s*(\d+)[ -]+(\d+)', scores_response)
            
            if not score1_match or not score2_match:
                print("❌ Première tentative pour les scores échouée. Réponse:")
                print(scores_response)
                
                # Deuxième tentative avec prompt ultra-simple
                retry_prompt = f"""Donne-moi exactement deux lignes au format suivant pour prédire le score de {match.home_team} vs {match.away_team}:

SCORE_1: 1-0
SCORE_2: 2-1

(Ces chiffres sont des exemples - utilise tes propres prédictions)"""

                retry_message = self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=50,
                    temperature=0.1,
                    messages=[{"role": "user", "content": retry_prompt}]
                )
                
                scores_response = retry_message.content[0].text.strip()
                score1_match = re.search(r'SCORE_1:?\s*(\d+)[ -]+(\d+)', scores_response)
                score2_match = re.search(r'SCORE_2:?\s*(\d+)[ -]+(\d+)', scores_response)
                
                # Si toujours pas de match, extraire simplement les paires de chiffres
                if not score1_match or not score2_match:
                    print("❌ Deuxième tentative pour les scores échouée. Réponse:")
                    print(scores_response)
                    
                    # Extraction de secours de n'importe quelles paires de chiffres
                    number_pairs = re.findall(r'(\d+)[ -]+(\d+)', scores_response)
                    if len(number_pairs) >= 2:
                        score1 = f"{number_pairs[0][0]}-{number_pairs[0][1]}"
                        score2 = f"{number_pairs[1][0]}-{number_pairs[1][1]}"
                        print(f"✅ Scores extraits du texte: {score1} et {score2}")
                    else:
                        # En dernier recours, utiliser des scores par défaut
                        score1 = "1-0"
                        score2 = "1-1"
                        print(f"⚠️ Utilisation de scores par défaut: {score1} et {score2}")
                else:
                    score1 = f"{score1_match.group(1)}-{score1_match.group(2)}"
                    score2 = f"{score2_match.group(1)}-{score2_match.group(2)}"
                    print(f"✅ Scores obtenus à la deuxième tentative: {score1} et {score2}")
            else:
                score1 = f"{score1_match.group(1)}-{score1_match.group(2)}"
                score2 = f"{score2_match.group(1)}-{score2_match.group(2)}"
                print(f"✅ Scores probables obtenus: {score1} et {score2}")
            
            # Enregistrer les scores dans l'objet match
            match.predicted_score1 = score1
            match.predicted_score2 = score2
            
            # Extraire les scores pour analyse ultérieure
            home_goals1, away_goals1 = map(int, score1.split('-'))
            home_goals2, away_goals2 = map(int, score2.split('-'))
            total_goals1 = home_goals1 + away_goals1
            total_goals2 = home_goals2 + away_goals2
            
            # ÉTAPE 2: Analyser et prédire le pari le plus sûr
            # Avec règles améliorées pour des prédictions plus fiables
            analysis_prompt = f"""Analyse ces données pour déterminer la prédiction LA PLUS SÛRE pour le match {match.home_team} vs {match.away_team}.

INFORMATIONS CLÉS:
- Scores probables: {score1} et {score2}
- Compétition: {match.competition} ({'Match de coupe' if match.is_cup_match else 'Match de championnat'})
- Buts totaux prévus: {total_goals1} et {total_goals2}

DONNÉES STATISTIQUES COMPLÈTES:
{data_content}

OPTIONS DISPONIBLES:
{', '.join(self.available_predictions)}

RÈGLES STRICTES DE SÉLECTION (TRÈS IMPORTANT):

1. PRIORISATION DES PRÉDICTIONS:
   - Priorité 1: Double chance (1X, X2) pour équipes favorites mais sans garantie
   - Priorité 2: Prédictions sur le nombre de buts (+1.5, +2.5, -3.5 buts)
   - Priorité 3: Résultat direct (1, 2) uniquement si confiance extrêmement élevée

2. RÈGLES POUR LES BUTS:
   - Pour "+1.5 buts": N'utiliser QUE si au moins 2 buts sont attendus dans CHAQUE score probable
   - Pour "+2.5 buts": N'utiliser QUE si au moins 3 buts sont attendus en moyenne
   - Pour "-3.5 buts": N'utiliser QUE si TOUS les scores prédits ont strictement moins de 4 buts

3. RÈGLES POUR LES DOUBLES CHANCES:
   - Pour "1X": L'équipe à domicile doit être légèrement favorite ou proche du nul
   - Pour "X2": L'équipe extérieure doit être légèrement favorite ou proche du nul
   - Ne JAMAIS utiliser "1X" si les deux scores favorisent l'équipe extérieure
   - Ne JAMAIS utiliser "X2" si les deux scores favorisent l'équipe à domicile

4. RÈGLES POUR LES VICTOIRES DIRECTES:
   - Pour "1": L'équipe à domicile doit gagner dans les DEUX scores probables
   - Pour "2": L'équipe extérieure doit gagner dans les DEUX scores probables
   - Exiger une confiance d'au moins 90% pour toute prédiction de résultat direct

5. COHÉRENCE AVEC LES SCORES:
   - La prédiction DOIT être cohérente avec les deux scores probables
   - En cas de scores contradictoires, privilégier les prédictions sur les buts

RÉPONDS EXACTEMENT AVEC CE FORMAT:
PRÉDICTION: [option choisie]
CONFIANCE: [pourcentage entre 85 et 100]"""

            prediction_message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,
                temperature=0.1,
                messages=[{"role": "user", "content": analysis_prompt}]
            )

            prediction_response = prediction_message.content[0].text.strip()
            
            # Extraction avec regex robuste
            prediction_match = re.search(r'PR[ÉE]DICTION:?\s*([^\n\r]+)', prediction_response)
            confidence_match = re.search(r'CONFIANCE:?\s*(\d+)', prediction_response)
            
            if not prediction_match or not confidence_match:
                print("❌ Première tentative pour la prédiction échouée. Réponse:")
                print(prediction_response)
                
                # Deuxième tentative avec prompt ultra-simple et pré-analyse des scores
                # Prédétermine la meilleure prédiction basée sur les scores
                suggested_pred = ""
                if home_goals1 > away_goals1 and home_goals2 > away_goals2:
                    # L'équipe à domicile gagne dans les deux scores
                    suggested_pred = "1X" if min(home_goals1 - away_goals1, home_goals2 - away_goals2) < 2 else "1"
                elif away_goals1 > home_goals1 and away_goals2 > home_goals2:
                    # L'équipe à l'extérieur gagne dans les deux scores
                    suggested_pred = "X2" if min(away_goals1 - home_goals1, away_goals2 - home_goals2) < 2 else "2"
                elif total_goals1 >= 3 or total_goals2 >= 3:
                    # Beaucoup de buts attendus
                    suggested_pred = "+2.5 buts"
                elif total_goals1 >= 2 or total_goals2 >= 2:
                    # Nombre modéré de buts
                    suggested_pred = "+1.5 buts"
                else:
                    # Match serré avec peu de buts
                    suggested_pred = "-3.5 buts"
                
                retry_prompt = f"""Choisis UNE SEULE prédiction parmi cette liste pour {match.home_team} vs {match.away_team}:
{', '.join(self.available_predictions)}

Basé sur les scores prédits {score1} et {score2}, la prédiction la plus sûre semble être {suggested_pred}.

Réponds UNIQUEMENT avec ces deux lignes:
PRÉDICTION: [ta prédiction finale]
CONFIANCE: [pourcentage entre 85 et 100]"""

                retry_message = self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=50,
                    temperature=0.1,
                    messages=[{"role": "user", "content": retry_prompt}]
                )
                
                prediction_response = retry_message.content[0].text.strip()
                prediction_match = re.search(r'PR[ÉE]DICTION:?\s*([^\n\r]+)', prediction_response)
                confidence_match = re.search(r'CONFIANCE:?\s*(\d+)', prediction_response)
                
                if not prediction_match or not confidence_match:
                    print("❌ Deuxième tentative pour la prédiction échouée")
                    # Utiliser notre suggestion en dernier recours
                    pred_text = suggested_pred
                    confidence = 85
                    print(f"⚠️ Utilisation de la prédiction de secours: {pred_text}")
                else:
                    pred_text = prediction_match.group(1).strip()
                    confidence = min(100, max(85, int(confidence_match.group(1))))
            else:
                pred_text = prediction_match.group(1).strip()
                confidence = min(100, max(85, int(confidence_match.group(1))))
            
            # Trouver la prédiction correspondante dans la liste disponible
            normalized_pred = None
            for available_pred in self.available_predictions:
                if available_pred.lower() in pred_text.lower():
                    normalized_pred = available_pred
                    break
            
            if not normalized_pred:
                print(f"❌ Prédiction '{pred_text}' non reconnue dans les options disponibles")
                
                # Extraction de secours - prendre la première prédiction qui apparaît dans le texte
                for available_pred in self.available_predictions:
                    if available_pred.lower() in prediction_response.lower():
                        normalized_pred = available_pred
                        print(f"✅ Prédiction extraite du texte: {normalized_pred}")
                        break
                
                if not normalized_pred:
                    # Dernier recours basé sur l'analyse des scores
                    if home_goals1 > away_goals1 and home_goals2 > away_goals2:
                        normalized_pred = "1X"
                    elif away_goals1 > home_goals1 and away_goals2 > home_goals2:
                        normalized_pred = "X2"
                    elif total_goals1 >= 3 or total_goals2 >= 3:
                        normalized_pred = "+2.5 buts"
                    else:
                        normalized_pred = "+1.5 buts"
                    
                    print(f"⚠️ Utilisation de prédiction par défaut basée sur les scores: {normalized_pred}")
                    if not normalized_pred:
                        return None
            
            # ÉTAPE 3: Vérification de cohérence entre les scores et la prédiction
            # Réduire la confiance ou changer la prédiction si nécessaire
            is_coherent = True
            
            # Vérifier si tous les scores prédits sont cohérents avec la prédiction
            if normalized_pred == "1" and (home_goals1 <= away_goals1 or home_goals2 <= away_goals2):
                is_coherent = False
                print("⚠️ Incohérence: prédiction '1' mais l'équipe à domicile ne gagne pas dans tous les scores")
            elif normalized_pred == "2" and (away_goals1 <= home_goals1 or away_goals2 <= home_goals2):
                is_coherent = False
                print("⚠️ Incohérence: prédiction '2' mais l'équipe extérieure ne gagne pas dans tous les scores")
            elif normalized_pred == "1X" and (away_goals1 > home_goals1 and away_goals2 > home_goals2):
                is_coherent = False
                print("⚠️ Incohérence: prédiction '1X' mais l'équipe extérieure est favorite dans tous les scores")
            elif normalized_pred == "X2" and (home_goals1 > away_goals1 and home_goals2 > away_goals2):
                is_coherent = False
                print("⚠️ Incohérence: prédiction 'X2' mais l'équipe à domicile est favorite dans tous les scores")
            elif normalized_pred == "+1.5 buts" and (total_goals1 < 2 and total_goals2 < 2):
                is_coherent = False
                print("⚠️ Incohérence: prédiction '+1.5 buts' mais moins de 2 buts sont attendus")
            elif normalized_pred == "+2.5 buts" and (total_goals1 < 3 and total_goals2 < 3):
                is_coherent = False
                print("⚠️ Incohérence: prédiction '+2.5 buts' mais moins de 3 buts sont attendus")
            elif normalized_pred == "-3.5 buts" and (total_goals1 >= 4 or total_goals2 >= 4):
                is_coherent = False
                print("⚠️ Incohérence: prédiction '-3.5 buts' mais 4 buts ou plus sont attendus")
            
            # Ajuster la prédiction si nécessaire
            if not is_coherent:
                print("🔄 Ajustement de la prédiction pour garantir la cohérence avec les scores...")
                
                # Proposer une prédiction alternative cohérente
                if normalized_pred in ["1", "1X"] and (away_goals1 > home_goals1 or away_goals2 > home_goals2):
                    # Les scores ne favorisent pas clairement l'équipe à domicile
                    if total_goals1 >= 3 or total_goals2 >= 3:
                        normalized_pred = "+2.5 buts"
                    else:
                        normalized_pred = "+1.5 buts"
                elif normalized_pred in ["2", "X2"] and (home_goals1 > away_goals1 or home_goals2 > away_goals2):
                    # Les scores ne favorisent pas clairement l'équipe extérieure
                    if total_goals1 >= 3 or total_goals2 >= 3:
                        normalized_pred = "+2.5 buts"
                    else:
                        normalized_pred = "+1.5 buts"
                elif normalized_pred == "+2.5 buts" and (total_goals1 < 3 and total_goals2 < 3):
                    normalized_pred = "+1.5 buts"
                elif normalized_pred == "+1.5 buts" and (total_goals1 < 2 and total_goals2 < 2):
                    if home_goals1 == away_goals1 or home_goals2 == away_goals2:  # Match nul probable
                        normalized_pred = "12"  # Double chance "Pas de match nul"
                    else:
                        normalized_pred = "-3.5 buts"
                
                print(f"✅ Nouvelle prédiction après ajustement: {normalized_pred}")
                
                # Réduire légèrement la confiance après ajustement
                confidence = max(85, confidence - 5)
            
            # Extraire les cotes pour la prédiction
            home_odds = match.bookmaker_odds.get("home", 0.0)
            draw_odds = match.bookmaker_odds.get("draw", 0.0)
            away_odds = match.bookmaker_odds.get("away", 0.0)
            
            # Créer l'objet de prédiction
            prediction = Prediction(
                region=match.region,
                competition=match.competition,
                match=f"{match.home_team} vs {match.away_team}",
                time=match.commence_time.astimezone(timezone(timedelta(hours=1))).strftime("%H:%M"),
                predicted_score1=score1,
                predicted_score2=score2,
                prediction=normalized_pred,
                confidence=confidence,
                home_odds=home_odds,
                draw_odds=draw_odds,
                away_odds=away_odds
            )
            
            print(f"✅ Prédiction finale: {normalized_pred} (Confiance: {confidence}%)")
            return prediction
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse avec Claude: {str(e)}")
            traceback.print_exc()
            return None

    def _format_predictions_message(self, predictions: List[Prediction]) -> str:
        # Date du jour formatée
        current_date = datetime.now().strftime('%d/%m/%Y')
        
        # En-tête du message avec formatage en gras
        msg = f"*🤖 AL VE AI BOT - PRÉDICTIONS DU {current_date} 🤖*\n\n"

        for i, pred in enumerate(predictions, 1):
            # Formatage des éléments avec gras et italique
            msg += (
                f"*📊 MATCH #{i}*\n"
                f"🏆 *{pred.competition}*\n"
                f"*⚔️ {pred.match}*\n"
                f"⏰ Coup d'envoi : *{pred.time}*\n"
                f"🔮 Scores prédits : *{pred.predicted_score1}* ou *{pred.predicted_score2}*\n"
                f"📈 Prédiction : *{pred.prediction}*\n"
                f"✅ Confiance : *{pred.confidence}%*\n\n"
                f"{'─' * 20}\n\n"
            )

        # Pied de page avec formatage en gras et italique
        msg += (
            "*⚠️ RAPPEL IMPORTANT :*\n"
            "• _Pariez de manière responsable_\n"
            "• _Ne dépassez pas 5% de votre bankroll_\n"
            "• *Ces prédictions sont basées sur l'analyse de données*"
        )
        return msg

    async def send_predictions(self, predictions: List[Prediction]) -> None:
        if not predictions:
            print("❌ Aucune prédiction à envoyer")
            return

        print("\n4️⃣ ENVOI DES PRÉDICTIONS")
        
        try:
            message = self._format_predictions_message(predictions)
            
            # Envoyer un message avec formatage Markdown
            await self.bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode="Markdown",  # Activer le formatage Markdown
                disable_web_page_preview=True
            )
            print(f"✅ {len(predictions)} prédictions envoyées!")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi des prédictions: {str(e)}")

    async def run(self) -> None:
        try:
            print(f"\n=== 🤖 AL VE AI BOT - GÉNÉRATION DES PRÉDICTIONS ({datetime.now().strftime('%H:%M')}) ===")
            
            # Étape 1: Récupérer les matchs en privilégiant les 5 grands championnats et leurs coupes
            all_matches = self.fetch_matches()
            if not all_matches:
                print("❌ Aucun match trouvé pour aujourd'hui")
                return

            predictions = []
            processed_count = 0
            
            # Étape 2: Analyser les matchs un par un jusqu'à obtenir assez de prédictions
            for match in all_matches:
                processed_count += 1
                
                # Si on a atteint le nombre maximum de prédictions, on s'arrête
                if len(predictions) >= self.config.MAX_MATCHES:
                    break
                
                match_type = "🏆 COUPE" if match.is_cup_match else "🏅 CHAMPIONNAT"
                print(f"\n📊 TRAITEMENT DU MATCH {processed_count}/{len(all_matches)}: {match_type} {match.home_team} vs {match.away_team}")
                
                # Collecter les données brutes via Perplexity
                data_collected = self.collect_match_data(match)
                if not data_collected:
                    print(f"⚠️ Données insuffisantes pour {match.home_team} vs {match.away_team}. Match ignoré.")
                    continue
                
                # Analyser le match avec Claude pour obtenir scores et prédiction
                prediction = self.analyze_with_claude(match)
                if prediction:
                    predictions.append(prediction)
                    print(f"✅ Prédiction {len(predictions)}/{self.config.MAX_MATCHES} obtenue")
                else:
                    print(f"⚠️ Aucune prédiction fiable pour {match.home_team} vs {match.away_team}")
                
                # Attendre un peu entre chaque analyse pour ne pas surcharger les API
                await asyncio.sleep(5)
            
            print(f"\n📊 {processed_count} matchs traités, {len(predictions)} prédictions obtenues")
            
            if predictions:
                if len(predictions) >= self.config.MIN_PREDICTIONS:
                    print(f"✅ Nombre requis de prédictions atteint: {len(predictions)}/{self.config.MIN_PREDICTIONS}")
                else:
                    print(f"⚠️ Seulement {len(predictions)}/{self.config.MIN_PREDICTIONS} prédictions obtenues")
                
                # Limiter au nombre maximum de prédictions si nécessaire
                if len(predictions) > self.config.MAX_MATCHES:
                    predictions = predictions[:self.config.MAX_MATCHES]
                
                # Envoyer les prédictions disponibles
                await self.send_predictions(predictions)
                print("=== ✅ EXÉCUTION TERMINÉE ===")
            else:
                print("❌ Aucune prédiction fiable n'a pu être générée")

        except Exception as e:
            print(f"❌ ERREUR GÉNÉRALE: {str(e)}")
            traceback.print_exc()

async def send_test_message(bot, chat_id):
    """Envoie un message de test pour vérifier la connectivité avec Telegram"""
    try:
        message = "*🤖 AL VE AI BOT - TEST DE CONNEXION*\n\nLe bot de paris a été déployé avec succès et est prêt à générer des prédictions!"
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown"
        )
        print("✅ Message de test envoyé")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi du message de test: {str(e)}")

async def run_once():
    """Exécute le bot une seule fois, pour les exécutions via Render cron job"""
    print("Démarrage du bot de paris sportifs en mode exécution unique...")
    
    # Configuration à partir des variables d'environnement
    config = Config(
        TELEGRAM_BOT_TOKEN=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        TELEGRAM_CHAT_ID=os.environ.get("TELEGRAM_CHAT_ID", ""),
        ODDS_API_KEY=os.environ.get("ODDS_API_KEY", ""),
        PERPLEXITY_API_KEY=os.environ.get("PERPLEXITY_API_KEY", ""),
        CLAUDE_API_KEY=os.environ.get("CLAUDE_API_KEY", ""),
        MAX_MATCHES=int(os.environ.get("MAX_MATCHES", "5")),
        MIN_PREDICTIONS=int(os.environ.get("MIN_PREDICTIONS", "5"))
    )
    
    bot = BettingBot(config)
    
    # Exécuter le bot directement sans message de test
    # pour éviter les erreurs si le chat_id est incorrect
    await bot.run()
    
    print("Exécution terminée.")

async def main():
    """Fonction principale qui détermine comment exécuter le bot"""
    try:
        print("Démarrage du bot de paris...")
        
        # Configuration à partir des variables d'environnement
        config = Config(
            TELEGRAM_BOT_TOKEN=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            TELEGRAM_CHAT_ID=os.environ.get("TELEGRAM_CHAT_ID", ""),
            ODDS_API_KEY=os.environ.get("ODDS_API_KEY", ""),
            PERPLEXITY_API_KEY=os.environ.get("PERPLEXITY_API_KEY", ""),
            CLAUDE_API_KEY=os.environ.get("CLAUDE_API_KEY", ""),
            MAX_MATCHES=int(os.environ.get("MAX_MATCHES", "5")),
            MIN_PREDICTIONS=int(os.environ.get("MIN_PREDICTIONS", "5"))
        )
        
        # Initialiser le bot
        bot = BettingBot(config)
        
        # Essayer d'envoyer un message de test, mais continuer même en cas d'échec
        try:
            await send_test_message(bot.bot, config.TELEGRAM_CHAT_ID)
        except Exception as e:
            print(f"⚠️ Message de test non envoyé: {str(e)}. Poursuite de l'exécution...")
        
        # Exécution immédiate
        print("⏰ Exécution immédiate au démarrage...")
        await bot.run()
        print("✅ Exécution immédiate terminée")
        
        # Initialiser la date du dernier jour d'exécution à aujourd'hui
        # pour éviter une nouvelle exécution le même jour
        today = datetime.now().day
        
        # Attendre jusqu'à 8h le lendemain
        print("🕒 Passage en mode attente: prochaine exécution planifiée à 8h00...")
        
        # Boucle principale du scheduler
        while True:
            # Heure actuelle en Afrique centrale (UTC+1)
            africa_central_tz = pytz.timezone("Africa/Lagos")  # Lagos est en UTC+1
            now = datetime.now(africa_central_tz)
            
            # Log d'activité toutes les heures (pour vérifier que le scheduler fonctionne)
            if now.minute == 0:
                print(f"Scheduler actif - Heure actuelle: {now.strftime('%Y-%m-%d %H:%M:%S')} (UTC+1)")
            
            # Exécution planifiée à 8h00, uniquement si on est un jour différent d'aujourd'hui
            if now.hour == 8 and now.minute < 10 and now.day != today:
                print(f"⏰ Exécution planifiée du bot à {now.strftime('%Y-%m-%d %H:%M:%S')} (heure d'Afrique centrale)")
                
                # Message de notification de début d'exécution
                try:
                    await bot.bot.send_message(
                        chat_id=config.TELEGRAM_CHAT_ID,
                        text="*⏰ GÉNÉRATION DES PRÉDICTIONS*\n\nLes prédictions du jour sont en cours de génération, veuillez patienter...",
                        parse_mode="Markdown"
                    )
                except Exception:
                    # Continuer même si le message ne peut pas être envoyé
                    print("⚠️ Message de notification non envoyé, poursuite de l'exécution...")
                
                # Exécuter le bot
                await bot.run()
                
                # Mettre à jour la date du jour après l'exécution
                today = now.day
                print(f"✅ Exécution terminée. Prochaine exécution prévue demain à 8h00")
                
                # Attendre un peu après l'exécution pour éviter les doublons
                await asyncio.sleep(600)  # 10 minutes
            
            # Vérifier toutes les 30 secondes
            await asyncio.sleep(30)
    
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE dans la fonction principale: {str(e)}")
        traceback.print_exc()
        
        # En cas d'erreur critique, attendre avant de quitter
        await asyncio.sleep(300)  # 5 minutes

if __name__ == "__main__":
    asyncio.run(main())
