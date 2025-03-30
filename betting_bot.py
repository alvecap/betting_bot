import asyncio
import requests
import anthropic
import logging
import telegram
import nest_asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
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
    stats: dict = None

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
        self.top_leagues = {
            # Championnats prioritaires (niveau 1)
            "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿": 1,
            "Championnat d'Espagne de Football 🇪🇸": 1,
            "Championnat d'Allemagne de Football 🇩🇪": 1,
            "Championnat d'Italie de Football 🇮🇹": 1,
            "Championnat de France de Football 🇫🇷": 1,
            "Ligue des Champions de l'UEFA 🇪🇺": 1,
            "Ligue Europa de l'UEFA 🇪🇺": 1,
            
            # Championnats secondaires (niveau 2)
            "Championnat de Belgique de Football 🇧🇪": 2,
            "Championnat des Pays-Bas de Football 🇳🇱": 2,
            "Championnat du Portugal de Football 🇵🇹": 2,
            "Premier League Russe 🇷🇺": 2,
            "Super League Suisse 🇨🇭": 2,
            "Süper Lig Turque 🇹🇷": 2,
            
            # Compétitions internationales (niveau 1)
            "Coupe du Monde FIFA 🌍": 1,
            "Ligue des Nations UEFA 🇪🇺": 1,
            "Qualifications Coupe du Monde UEFA 🇪🇺": 1,
            "Qualifications Coupe du Monde CAF 🌍": 1,
            "Qualifications Coupe du Monde CONCACAF 🌎": 1,
            "Qualifications Coupe du Monde CONMEBOL 🌎": 1,
            "Qualifications Coupe du Monde AFC 🌏": 1,
            "Qualifications Coupe du Monde OFC 🌏": 1,
            "Coupe d'Afrique des Nations 🌍": 1,
            "Copa America 🌎": 1,
            "Championnat d'Europe UEFA 🇪🇺": 1,
            
            # Autres championnats internationaux (niveau 3)
            "MLS 🇺🇸": 3,
            "Liga MX 🇲🇽": 3,
            "J-League 🇯🇵": 3,
            "K-League 🇰🇷": 3,
            "A-League 🇦🇺": 3,
            "Chinese Super League 🇨🇳": 3,
            "Brasileirão 🇧🇷": 3,
            "Argentine Primera División 🇦🇷": 3
        }
        print("Bot initialisé!")

    def _get_league_name(self, competition: str) -> str:
        league_mappings = {
            # Grands championnats européens
            "Premier League": "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "La Liga": "Championnat d'Espagne de Football 🇪🇸",
            "Bundesliga": "Championnat d'Allemagne de Football 🇩🇪",
            "Serie A": "Championnat d'Italie de Football 🇮🇹",
            "Ligue 1": "Championnat de France de Football 🇫🇷",
            
            # Coupes européennes
            "Champions League": "Ligue des Champions de l'UEFA 🇪🇺",
            "Europa League": "Ligue Europa de l'UEFA 🇪🇺",
            "Conference League": "Ligue Conférence de l'UEFA 🇪🇺",
            
            # Championnats européens secondaires
            "Belgian First Division A": "Championnat de Belgique de Football 🇧🇪",
            "Eredivisie": "Championnat des Pays-Bas de Football 🇳🇱",
            "Primeira Liga": "Championnat du Portugal de Football 🇵🇹",
            "Russian Premier League": "Premier League Russe 🇷🇺",
            "Swiss Super League": "Super League Suisse 🇨🇭",
            "Turkish Super Lig": "Süper Lig Turque 🇹🇷",
            
            # Compétitions internationales
            "FIFA World Cup": "Coupe du Monde FIFA 🌍",
            "UEFA Nations League": "Ligue des Nations UEFA 🇪🇺",
            "UEFA European Championship": "Championnat d'Europe UEFA 🇪🇺",
            "FIFA World Cup Qualification (UEFA)": "Qualifications Coupe du Monde UEFA 🇪🇺",
            "FIFA World Cup Qualification (CAF)": "Qualifications Coupe du Monde CAF 🌍",
            "FIFA World Cup Qualification (CONCACAF)": "Qualifications Coupe du Monde CONCACAF 🌎",
            "FIFA World Cup Qualification (CONMEBOL)": "Qualifications Coupe du Monde CONMEBOL 🌎",
            "FIFA World Cup Qualification (AFC)": "Qualifications Coupe du Monde AFC 🌏",
            "FIFA World Cup Qualification (OFC)": "Qualifications Coupe du Monde OFC 🌏",
            "Africa Cup of Nations": "Coupe d'Afrique des Nations 🌍",
            "Copa America": "Copa America 🌎",
            
            # Autres championnats internationaux
            "Major League Soccer": "MLS 🇺🇸",
            "Liga MX": "Liga MX 🇲🇽",
            "J League": "J-League 🇯🇵",
            "K League 1": "K-League 🇰🇷",
            "A-League": "A-League 🇦🇺",
            "Chinese Super League": "Chinese Super League 🇨🇳",
            "Brasileirão": "Brasileirão 🇧🇷",
            "Argentine Primera División": "Argentine Primera División 🇦🇷"
        }
        return league_mappings.get(competition, competition)

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def fetch_matches(self, max_match_count: int = 30) -> List[Match]:
        """Récupère les matchs à venir en priorisant les meilleures ligues"""
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
            print(f"✅ {len(matches_data)} matchs récupérés")

            current_time = datetime.now(timezone.utc)
            all_matches = []

            # Collecter tous les matchs des prochaines 48h
            for match_data in matches_data:
                commence_time = datetime.fromisoformat(match_data["commence_time"].replace('Z', '+00:00'))
                competition = self._get_league_name(match_data.get("sport_title", "Unknown"))
                
                # Filtrer sur les prochaines 48h
                if 0 < (commence_time - current_time).total_seconds() <= 172800:  # 48 heures
                    all_matches.append(Match(
                        home_team=match_data["home_team"],
                        away_team=match_data["away_team"],
                        competition=competition,
                        region=competition.split()[-1] if " " in competition else competition,
                        commence_time=commence_time,
                        priority=self.top_leagues.get(competition, 4)  # Priorité 4 par défaut (la plus basse)
                    ))

            if not all_matches:
                print("❌ Aucun match trouvé pour les prochaines 48 heures")
                return []

            # Trier les matchs par priorité (les plus importantes d'abord)
            all_matches.sort(key=lambda x: (x.priority, x.commence_time))
            
            # Calcul du nombre de matchs à sélectionner par niveau de priorité
            total_required = self.config.MIN_PREDICTIONS
            
            # Sélection des meilleurs matchs selon la priorité
            priority1_matches = [m for m in all_matches if m.priority == 1]
            priority2_matches = [m for m in all_matches if m.priority == 2]
            priority3_matches = [m for m in all_matches if m.priority == 3]
            other_matches = [m for m in all_matches if m.priority > 3]
            
            selected_matches = []
            
            # Priorité 1: prendre au moins 60% des matchs si disponible
            if priority1_matches:
                num_p1 = min(int(total_required * 0.6) + 1, len(priority1_matches))
                selected_matches.extend(random.sample(priority1_matches, num_p1))
            
            # Priorité 2: compléter jusqu'à 80% du total
            remaining_for_p2 = int(total_required * 0.8) - len(selected_matches)
            if remaining_for_p2 > 0 and priority2_matches:
                num_p2 = min(remaining_for_p2, len(priority2_matches))
                selected_matches.extend(random.sample(priority2_matches, num_p2))
            
            # Priorité 3: compléter jusqu'à 95% du total
            remaining_for_p3 = int(total_required * 0.95) - len(selected_matches)
            if remaining_for_p3 > 0 and priority3_matches:
                num_p3 = min(remaining_for_p3, len(priority3_matches))
                selected_matches.extend(random.sample(priority3_matches, num_p3))
            
            # Autres matchs: compléter si nécessaire
            remaining_needed = total_required - len(selected_matches)
            if remaining_needed > 0 and other_matches:
                num_other = min(remaining_needed, len(other_matches))
                selected_matches.extend(random.sample(other_matches, num_other))
            
            # Si on n'a toujours pas assez, reprendre des matchs prioritaires
            if len(selected_matches) < total_required:
                remaining = [m for m in all_matches if m not in selected_matches]
                if remaining:
                    still_needed = total_required - len(selected_matches)
                    selected_matches.extend(random.sample(remaining, min(still_needed, len(remaining))))
            
            print(f"\n✅ {len(selected_matches)} matchs candidats sélectionnés:")
            for i, match in enumerate(selected_matches, 1):
                print(f"  {i}. {match.home_team} vs {match.away_team} ({match.competition}, Priorité: {match.priority})")
                
            return selected_matches

        except Exception as e:
            print(f"❌ Erreur lors de la récupération des matchs: {str(e)}")
            return []

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def collect_match_data(self, match: Match) -> Optional[dict]:
        """Collecte toutes les données brutes nécessaires pour l'analyse du match via Perplexity"""
        print(f"\n2️⃣ COLLECTE DE DONNÉES POUR {match.home_team} vs {match.away_team}")
        try:
            # Structure pour collecter les différents types de données
            match_data = {
                "forme_recente": None,
                "confrontations_directes": None,
                "statistiques_detaillees": None,
                "absences_effectif": None,
                "contexte_match": None
            }
            
            # 1. Forme récente
            forme_prompt = f"""Tu es un collecteur de données sportives factuel. Fournir UNIQUEMENT ces statistiques précises et fiables pour {match.home_team} et {match.away_team}:

1. Les 5 derniers matchs de chaque équipe au format: Date | Compétition | Match | Score
2. La forme actuelle (ex: VVNDV)
3. Moyenne de buts marqués et encaissés sur les 5 derniers matchs
4. Tendance offensive et défensive

IMPORTANT: Donne UNIQUEMENT les statistiques brutes sans aucune analyse ni conclusion. Format sous forme de liste."""

            forme_response = self._get_perplexity_response(forme_prompt)
            if forme_response:
                match_data["forme_recente"] = forme_response
                print("✅ Données de forme récente collectées")
            
            # 2. Confrontations directes
            h2h_prompt = f"""En tant que collecteur de données sportives, fournir UNIQUEMENT les résultats des 5 dernières confrontations directes entre {match.home_team} et {match.away_team}:

Format pour chaque match: Date | Compétition | Match | Score

Ajoute également:
- Bilan global: X victoires pour {match.home_team}, Y victoires pour {match.away_team}, Z nuls
- Nombre moyen de buts par match lors des confrontations directes

IMPORTANT: Donne UNIQUEMENT les données brutes sans interprétation."""

            h2h_response = self._get_perplexity_response(h2h_prompt)
            if h2h_response:
                match_data["confrontations_directes"] = h2h_response
                print("✅ Données de confrontations directes collectées")
            
            # 3. Statistiques détaillées
            stats_prompt = f"""En tant que collecteur de données sportives, fournir uniquement ces statistiques précises pour {match.home_team} et {match.away_team}:

1. Pourcentage exact de matchs avec +1.5 buts cette saison
2. Pourcentage exact de matchs avec +2.5 buts cette saison
3. Pourcentage exact de matchs avec -3.5 buts cette saison
4. Pourcentage de matchs où les deux équipes marquent
5. Pourcentage de clean sheets (matchs sans encaisser de but)
6. Statistiques à domicile/extérieur (selon l'équipe)
7. Buts par mi-temps (1ère/2ème) cette saison

IMPORTANT: Données précises et factuelles uniquement, format tableau ou liste."""

            stats_response = self._get_perplexity_response(stats_prompt)
            if stats_response:
                match_data["statistiques_detaillees"] = stats_response
                print("✅ Données statistiques détaillées collectées")
            
            # 4. Absences et effectif
            absences_prompt = f"""En tant que collecteur de données sportives, fournir uniquement ces informations factuelles sur les effectifs:

1. Liste des joueurs blessés ou suspendus pour {match.home_team}
2. Liste des joueurs blessés ou suspendus pour {match.away_team}
3. Joueurs clés de retour récemment
4. État de forme des buteurs principaux (buts récents)

IMPORTANT: Format liste, données factuelles uniquement sans analyse."""

            absences_response = self._get_perplexity_response(absences_prompt)
            if absences_response:
                match_data["absences_effectif"] = absences_response
                print("✅ Données sur les absences et effectifs collectées")
            
            # 5. Contexte du match
            contexte_prompt = f"""En tant que collecteur de données sportives, fournir uniquement ces informations factuelles sur le contexte du match {match.home_team} vs {match.away_team} ({match.competition}):

1. Position actuelle au classement des deux équipes
2. Enjeu sportif exact (qualification, relégation, etc.)
3. Importance du match dans le calendrier
4. Conditions extérieures prévues (météo, état du terrain)
5. Affluence attendue et ambiance

IMPORTANT: Données factuelles uniquement, pas d'analyse ni d'opinion."""

            contexte_response = self._get_perplexity_response(contexte_prompt)
            if contexte_response:
                match_data["contexte_match"] = contexte_response
                print("✅ Données sur le contexte du match collectées")
            
            # Vérifier que nous avons au moins les données essentielles
            if match_data["forme_recente"] and match_data["statistiques_detaillees"]:
                print("✅ Données suffisantes collectées pour l'analyse")
                return match_data
            else:
                print("❌ Données insuffisantes pour l'analyse")
                return None
                
        except Exception as e:
            print(f"❌ Erreur lors de la collecte des données: {str(e)}")
            return None

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

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def analyze_with_claude(self, match: Match) -> Optional[Tuple[Prediction, Tuple[str, str]]]:
        """Analyse complète du match avec Claude pour obtenir les scores probables et la prédiction"""
        print(f"\n3️⃣ ANALYSE AVEC CLAUDE POUR {match.home_team} vs {match.away_team}")
        
        if not match.stats:
            print("❌ Aucune donnée statistique disponible pour l'analyse")
            return None
        
        try:
            # Préparer les données pour Claude
            data_sections = []
            for section_name, content in match.stats.items():
                if content:
                    formatted_section = f"### {section_name.upper().replace('_', ' ')}\n{content}"
                    data_sections.append(formatted_section)
            
            data_content = "\n\n".join(data_sections)
            
            # Étape 1: Obtenir les scores probables
            scores_prompt = f"""Tu es un expert en prédiction de scores exacts pour les matchs de football, utilisant une approche factuelle et statistique.

# INFORMATIONS SUR LE MATCH
Match: {match.home_team} vs {match.away_team}
Compétition: {match.competition}
Date: {match.commence_time.strftime('%d/%m/%Y')}

# DONNÉES STATISTIQUES COMPLÈTES
{data_content}

# TÂCHE
En utilisant UNIQUEMENT les données statistiques ci-dessus et ton expertise en modélisation statistique:

1. Utilise une méthode ELO avancée pour évaluer la force relative des équipes
2. Applique un modèle de Poisson pour estimer la distribution probable des buts
3. Analyse l'impact des facteurs contextuels (absences, enjeu, forme)
4. Considère les tendances des confrontations directes
5. Évalue séparément les performances à domicile/extérieur
6. Produis DEUX scores exacts les plus probables pour ce match

# FORMAT DE RÉPONSE REQUIS
réponds UNIQUEMENT au format suivant:
SCORE_1: X-Y
SCORE_2: Z-W

où X, Y, Z et W sont des nombres entiers représentant les scores les plus probables selon ton analyse statistique.
"""

            scores_message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.1,
                messages=[{"role": "user", "content": scores_prompt}]
            )

            scores_response = scores_message.content[0].text
            
            # Extraire les deux scores
            score1_match = re.search(r'SCORE_1:\s*(\d+)-(\d+)', scores_response)
            score2_match = re.search(r'SCORE_2:\s*(\d+)-(\d+)', scores_response)
            
            if score1_match and score2_match:
                score1 = f"{score1_match.group(1)}-{score1_match.group(2)}"
                score2 = f"{score2_match.group(1)}-{score2_match.group(2)}"
                print(f"✅ Scores probables obtenus: {score1} et {score2}")
                
                # Étape 2: Analyser et prédire le pari le plus sûr
                prediction_prompt = f"""Tu es un expert en analyse de paris sportifs qui fait des recommandations basées uniquement sur les données statistiques et les scores probables.

# INFORMATIONS SUR LE MATCH
Match: {match.home_team} vs {match.away_team}
Compétition: {match.competition}
Date: {match.commence_time.strftime('%d/%m/%Y')}
Scores probables: {score1} et {score2}

# DONNÉES STATISTIQUES COMPLÈTES
{data_content}

# OPTIONS DE PRÉDICTION DISPONIBLES
{', '.join(self.available_predictions)}

# RÈGLES STRICTES POUR LA SÉLECTION DE PRÉDICTION
1. Ignore complètement la réputation ou la notoriété des équipes
2. Base ta prédiction UNIQUEMENT sur les données statistiques et les scores probables
3. La prédiction doit être cohérente avec les scores probables
4. Ne recommande une victoire directe (1 ou 2) que si ton niveau de confiance est d'au moins 90%
5. Ne recommande pas "1X" si les scores probables favorisent l'équipe extérieure
6. Ne recommande pas "X2" si les scores probables favorisent l'équipe à domicile
7. Pour "+1.5 buts", assure-toi que tu es sûr à 90% qu'il y aura au moins 2 buts
8. Pour "+2.5 buts", assure-toi que tu es sûr à 90% qu'il y aura au moins 3 buts
9. Pour "-3.5 buts", assure-toi que tu es sûr à 85% qu'il y aura moins de 4 buts
10. Ta confiance minimum pour toute prédiction doit être d'au moins 85%

# FORMAT DE RÉPONSE REQUIS
PRÉDICTION: [une seule option parmi la liste]
CONFIANCE: [pourcentage précis entre 85 et 100]

N'inclus aucune explication ou justification, seulement ces deux lignes.
"""

                prediction_message = self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=300,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prediction_prompt}]
                )

                prediction_response = prediction_message.content[0].text
                
                prediction_match = re.search(r'PRÉDICTION:\s*(.*)', prediction_response)
                confidence_match = re.search(r'CONFIANCE:\s*(\d+)', prediction_response)
                
                if prediction_match and confidence_match:
                    pred = prediction_match.group(1).strip()
                    conf = min(100, max(85, int(confidence_match.group(1))))
                    
                    # Normaliser la prédiction au format exact souhaité
                    normalized_pred = None
                    for available in self.available_predictions:
                        if available.lower() in pred.lower():
                            normalized_pred = available
                            break
                    
                    if normalized_pred:
                        prediction = Prediction(
                            region=match.region,
                            competition=match.competition,
                            match=f"{match.home_team} vs {match.away_team}",
                            time=match.commence_time.astimezone(timezone(timedelta(hours=1))).strftime("%H:%M"),
                            predicted_score1=score1,
                            predicted_score2=score2,
                            prediction=normalized_pred,
                            confidence=conf
                        )
                        
                        print(f"✅ Prédiction obtenue: {normalized_pred} (Confiance: {conf}%)")
                        return prediction, (score1, score2)
                    else:
                        print(f"❌ Prédiction {pred} non reconnue parmi les options disponibles")
                else:
                    print("❌ Format de prédiction invalide")
            else:
                print("❌ Format de scores invalide")
            
            return None
                
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse avec Claude: {str(e)}")
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
            
            # Étape 1: Récupérer les matchs en privilégiant les compétitions importantes
            all_matches = self.fetch_matches(max_match_count=30)
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
                
                print(f"\n📊 TRAITEMENT DU MATCH {processed_count}/{len(all_matches)}: {match.home_team} vs {match.away_team}")
                
                # Collecter les données brutes via Perplexity
                match.stats = self.collect_match_data(match)
                if not match.stats:
                    print(f"⚠️ Données insuffisantes pour {match.home_team} vs {match.away_team}. Match ignoré.")
                    continue
                
                # Analyser le match avec Claude pour obtenir scores et prédiction
                analysis_result = self.analyze_with_claude(match)
                if analysis_result:
                    prediction, scores = analysis_result
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
    
    # Envoyer un message de test
    await send_test_message(bot.bot, config.TELEGRAM_CHAT_ID)
    
    # Exécuter le bot
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
        
        # Test de connexion
        await send_test_message(bot.bot, config.TELEGRAM_CHAT_ID)
        
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
                await bot.bot.send_message(
                    chat_id=config.TELEGRAM_CHAT_ID,
                    text="*⏰ GÉNÉRATION DES PRÉDICTIONS*\n\nLes prédictions du jour sont en cours de génération, veuillez patienter...",
                    parse_mode="Markdown"
                )
                
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
