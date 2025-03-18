import asyncio
import requests
import anthropic
import logging
import telegram
import nest_asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import sys
from retry import retry
import pytz  # Pour gérer les fuseaux horaires
import os   # Pour les variables d'environnement
import random  # Pour sélectionner des matchs supplémentaires si nécessaire

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
    MIN_PREDICTIONS: int = 5  # Nombre minimum de prédictions à collecter

@dataclass
class Match:
    home_team: str
    away_team: str
    competition: str
    region: str
    commence_time: datetime
    bookmakers: List[Dict]
    all_odds: List[Dict]
    priority: int = 0
    predicted_score1: str = ""
    predicted_score2: str = ""
    home_odds: float = 0.0
    draw_odds: float = 0.0
    away_odds: float = 0.0

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
            "1", "2",  # Suppression du match nul comme prédiction
            "-1.5 buts 1ère mi-temps", 
            "+0.5 but 1ère mi-temps", "+0.5 but 2ème mi-temps"
        ]
        self.top_leagues = {
            "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿": 1,
            "Championnat d'Espagne de Football 🇪🇸": 1,
            "Championnat d'Allemagne de Football 🇩🇪": 1,
            "Championnat d'Italie de Football 🇮🇹": 1,
            "Championnat de France de Football 🇫🇷": 1,
            "Ligue des Champions de l'UEFA 🇪🇺": 1,
            "Ligue Europa de l'UEFA 🇪🇺": 1,
            "Championnat de Belgique de Football 🇧🇪": 2,
            "Championnat des Pays-Bas de Football 🇳🇱": 2,
            "Championnat du Portugal de Football 🇵🇹": 2
        }
        print("Bot initialisé!")

    def _get_league_name(self, competition: str) -> str:
        league_mappings = {
            "Premier League": "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "La Liga": "Championnat d'Espagne de Football 🇪🇸",
            "Bundesliga": "Championnat d'Allemagne de Football 🇩🇪",
            "Serie A": "Championnat d'Italie de Football 🇮🇹",
            "Ligue 1": "Championnat de France de Football 🇫🇷",
            "Champions League": "Ligue des Champions de l'UEFA 🇪🇺",
            "Europa League": "Ligue Europa de l'UEFA 🇪🇺",
            "Belgian First Division A": "Championnat de Belgique de Football 🇧🇪",
            "Dutch Eredivisie": "Championnat des Pays-Bas de Football 🇳🇱",
            "Primeira Liga": "Championnat du Portugal de Football 🇵🇹"
        }
        return league_mappings.get(competition, competition)

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def fetch_matches(self, max_match_count: int = 15) -> List[Match]:
        """Récupère plus de matchs que nécessaire pour avoir des alternatives si certains échouent"""
        print("\n1️⃣ RÉCUPÉRATION DES MATCHS...")
        url = "https://api.the-odds-api.com/v4/sports/soccer/odds/"
        params = {
            "apiKey": self.config.ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            matches_data = response.json()
            print(f"✅ {len(matches_data)} matchs récupérés")

            current_time = datetime.now(timezone.utc)
            matches = []

            for match_data in matches_data:
                commence_time = datetime.fromisoformat(match_data["commence_time"].replace('Z', '+00:00'))
                # Prendre les matchs des prochaines 24 heures
                if 0 < (commence_time - current_time).total_seconds() <= 86400:
                    competition = self._get_league_name(match_data.get("sport_title", "Unknown"))
                    
                    # Extraire les cotes des bookmakers
                    home_odds, draw_odds, away_odds = 0.0, 0.0, 0.0
                    
                    if match_data.get("bookmakers") and len(match_data["bookmakers"]) > 0:
                        for bookmaker in match_data["bookmakers"]:
                            if bookmaker.get("markets") and len(bookmaker["markets"]) > 0:
                                for market in bookmaker["markets"]:
                                    if market["key"] == "h2h" and len(market["outcomes"]) == 3:
                                        for outcome in market["outcomes"]:
                                            if outcome["name"] == match_data["home_team"]:
                                                home_odds = outcome["price"]
                                            elif outcome["name"] == match_data["away_team"]:
                                                away_odds = outcome["price"]
                                            else:
                                                draw_odds = outcome["price"]
                                        break
                            if home_odds > 0 and draw_odds > 0 and away_odds > 0:
                                break
                    
                    matches.append(Match(
                        home_team=match_data["home_team"],
                        away_team=match_data["away_team"],
                        competition=competition,
                        region=competition.split()[-1] if " " in competition else competition,
                        commence_time=commence_time,
                        bookmakers=match_data.get("bookmakers", []),
                        all_odds=match_data.get("bookmakers", []),
                        priority=self.top_leagues.get(competition, 0),
                        home_odds=home_odds,
                        draw_odds=draw_odds,
                        away_odds=away_odds
                    ))

            if not matches:
                return []

            # Trier les matchs par priorité et heure de début
            matches.sort(key=lambda x: (-x.priority, x.commence_time))
            
            # Prendre plus de matchs que nécessaire pour avoir des alternatives
            top_matches = matches[:max_match_count]
            
            print(f"\n✅ {len(top_matches)} matchs candidats sélectionnés")
            for match in top_matches[:5]:
                print(f"- {match.home_team} vs {match.away_team} ({match.competition}) - Cotes: {match.home_odds}/{match.draw_odds}/{match.away_odds}")
                
            return top_matches

        except Exception as e:
            print(f"❌ Erreur lors de la récupération des matchs: {str(e)}")
            return []

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def get_match_stats(self, match: Match) -> Optional[str]:
        print(f"\n2️⃣ ANALYSE DE {match.home_team} vs {match.away_team}")
        try:
            # Utiliser le même modèle et prompt que pour les scores exacts
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [{
                        "role": "user", 
                        "content": f"""Tu es une intelligence artificielle experte en analyse sportive de football. 

Fais une analyse détaillée et factuelle pour le match {match.home_team} vs {match.away_team} ({match.competition}) qui aura lieu le {match.commence_time.strftime('%d/%m/%Y')}.

Analyse OBLIGATOIREMENT tous ces éléments:
1. FORME RÉCENTE:
   - 5 derniers matchs de chaque équipe avec les résultats exacts
   - Moyenne de buts marqués/encaissés par match
   - Performance à domicile/extérieur (pourcentage de victoires)

2. CONFRONTATIONS DIRECTES:
   - Les 5 dernières rencontres entre ces équipes avec scores
   - Tendances des confrontations (équipe dominante)
   - Nombre moyen de buts dans ces confrontations

3. STATISTIQUES CLÉS:
   - Pourcentage de matchs avec +1.5 buts pour chaque équipe
   - Pourcentage de matchs avec +2.5 buts pour chaque équipe
   - Pourcentage de matchs avec -3.5 buts pour chaque équipe
   - Pourcentage de matchs où les deux équipes marquent

4. ABSENCES ET EFFECTIF:
   - Joueurs blessés ou suspendus importants
   - Impact des absences sur le jeu de l'équipe

5. CONTEXTE DU MATCH:
   - Enjeu sportif (qualification, maintien, position au classement)
   - Importance du match pour chaque équipe

Sois aussi précis et factuel que possible avec des statistiques réelles."""
                    }],
                    "max_tokens": 700,
                    "temperature": 0.1
                },
                timeout=60  # 1 minute pour obtenir les statistiques
            )
            response.raise_for_status()
            stats = response.json()["choices"][0]["message"]["content"]
            print("✅ Statistiques complètes récupérées")
            return stats
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des statistiques: {str(e)}")
            
            # En cas d'échec, essayer avec un prompt plus court
            try:
                print("⚠️ Tentative avec un prompt simplifié...")
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={"Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                            "Content-Type": "application/json"},
                    json={
                        "model": "sonar",
                        "messages": [{
                            "role": "user", 
                            "content": f"""Analyse factuelle pour le match {match.home_team} vs {match.away_team} ({match.competition}):

1. Forme récente des deux équipes (résultats des 5 derniers matchs)
2. Confrontations directes récentes
3. Statistiques: matchs avec +1.5 buts, +2.5 buts, -3.5 buts
4. Absences importantes
5. Enjeu du match"""
                        }],
                        "max_tokens": 500,
                        "temperature": 0.1
                    },
                    timeout=45
                )
                response.raise_for_status()
                stats = response.json()["choices"][0]["message"]["content"]
                print("✅ Statistiques basiques récupérées")
                return stats
            except Exception as e:
                print(f"❌ Erreur lors de la récupération des statistiques simplifiées: {str(e)}")
                return None

    @retry(tries=2, delay=5, backoff=2, logger=logger)
    def get_predicted_scores(self, match: Match) -> Optional[tuple]:
        """Récupère les scores prédits, retourne None si impossible d'obtenir des prédictions fiables"""
        print(f"\n3️⃣ OBTENTION DES SCORES EXACTS PROBABLES POUR {match.home_team} vs {match.away_team}")
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [{
                        "role": "user", 
                        "content": f"""Tu es une intelligence artificielle experte en paris sportifs, spécialisée dans la prédiction de scores exacts. Tu utilises des modèles statistiques avancés, y compris la méthode ELO, pour évaluer la force relative des équipes et estimer le nombre de buts potentiels de chaque équipe dans un match.

Objectif: Générer deux propositions de scores exacts pour le match {match.home_team} vs {match.away_team} qui aura lieu le {match.commence_time.strftime('%d/%m/%Y')} en {match.competition}.

Pour générer ces prédictions, analyse les éléments suivants:
1. Contexte du match (compétition, enjeu, phase du tournoi)
2. Forme et performances des équipes (5 derniers matchs, buts marqués/encaissés)
3. Confrontations directes (historique entre les équipes)
4. Absences et forme des joueurs clés
5. Analyse avec la méthode ELO et statistiques avancées
6. Tendances des bookmakers et experts
7. Facteurs psychologiques et extra-sportifs

Réponds UNIQUEMENT au format "Score 1: X-Y, Score 2: Z-W" où X,Y,Z,W sont des nombres entiers. Ne donne aucune autre information ou explication."""
                    }],
                    "max_tokens": 100,
                    "temperature": 0.1
                },
                timeout=120  # 2 minutes pour les scores exacts
            )
            response.raise_for_status()
            prediction_text = response.json()["choices"][0]["message"]["content"].strip()
            
            # Extraire les deux scores
            score1_match = re.search(r'Score 1:\s*(\d+)-(\d+)', prediction_text)
            score2_match = re.search(r'Score 2:\s*(\d+)-(\d+)', prediction_text)
            
            if score1_match and score2_match:
                score1 = f"{score1_match.group(1)}-{score1_match.group(2)}"
                score2 = f"{score2_match.group(1)}-{score2_match.group(2)}"
                print(f"✅ Scores probables obtenus: {score1} et {score2}")
                return score1, score2
            else:
                # Si le format n'est pas respecté, extraire les scores si possible
                scores = re.findall(r'(\d+)[^\d]+(\d+)', prediction_text)
                if len(scores) >= 2:
                    score1 = f"{scores[0][0]}-{scores[0][1]}"
                    score2 = f"{scores[1][0]}-{scores[1][1]}"
                    print(f"✅ Scores probables extraits: {score1} et {score2}")
                    return score1, score2
                else:
                    print("❌ Format de scores invalide, match ignoré")
                    return None
                
        except Exception as e:
            print(f"❌ Erreur lors de l'obtention des scores probables: {str(e)}")
            return None

    def analyze_match(self, match: Match, stats: str) -> Optional[Prediction]:
        print(f"\n4️⃣ ANALYSE AVEC CLAUDE POUR {match.home_team} vs {match.away_team}")
        
        try:
            prompt = f"""ANALYSE APPROFONDIE: {match.home_team} vs {match.away_team}
COMPÉTITION: {match.competition}
SCORES EXACTS PRÉDITS: {match.predicted_score1} et {match.predicted_score2}
COTES: Victoire {match.home_team}: {match.home_odds}, Match nul: {match.draw_odds}, Victoire {match.away_team}: {match.away_odds}

DONNÉES STATISTIQUES:
{stats}

CONSIGNES STRICTES:
1. Analyser en profondeur les statistiques fournies et les scores exacts prédits
2. Évaluer les tendances et performances des équipes
3. Considérer les scores exacts prédits par les experts
4. Choisir la prédiction LA PLUS SÛRE parmi: {', '.join(self.available_predictions)}

RÈGLES DE VÉRIFICATION OBLIGATOIRES:
- Pour prédire une victoire à domicile "1", l'équipe à domicile doit avoir une cote MAXIMALE de 1.50
- Pour prédire une victoire à l'extérieur "2", l'équipe extérieure doit avoir une cote MAXIMALE de 1.50
- Si la cote est supérieure à 1.50, NE PAS prédire de victoire directe; préférer double chance (1X ou X2)
- Pour prédire "+1.5 buts", on doit être sûr à 90% que le match aura AU MOINS 3 BUTS
- Pour prédire "+2.5 buts", on doit être sûr à 90% que le match aura AU MOINS 4 BUTS
- Pour prédire "-3.5 buts", la probabilité doit être d'au moins 80% que le match aura moins de 4 buts
- Ne jamais donner de prédiction sans une confiance d'au moins 80%
- Le match nul "X" n'est PAS une option de prédiction acceptable
- Privilégier les prédictions avec les statistiques les plus solides

FORMAT REQUIS:
PREDICTION: [une option de la liste]
CONFIANCE: [pourcentage]"""

            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text
            prediction = re.search(r"PREDICTION:\s*(.*)", response)
            confidence = re.search(r"CONFIANCE:\s*(\d+)", response)

            if all([prediction, confidence]):
                pred = prediction.group(1).strip()
                conf = min(100, max(80, int(confidence.group(1))))
                
                if any(p.lower() in pred.lower() for p in self.available_predictions):
                    # Trouver la prédiction exacte dans la liste
                    for available_pred in self.available_predictions:
                        if available_pred.lower() in pred.lower():
                            pred = available_pred
                            break
                    
                    # Vérifications supplémentaires pour la fiabilité
                    # Appliquer les règles strictes pour les victoires directes
                    if pred == "1" and match.home_odds > 1.50:
                        print(f"⚠️ Cote domicile trop élevée ({match.home_odds} > 1.50). Conversion en 1X.")
                        pred = "1X"
                    elif pred == "2" and match.away_odds > 1.50:
                        print(f"⚠️ Cote extérieur trop élevée ({match.away_odds} > 1.50). Conversion en X2.")
                        pred = "X2"
                    
                    if pred == "X":
                        print("⚠️ Prédiction de match nul non autorisée. Prédiction rejetée.")
                        return None
                    
                    print(f"✅ Prédiction finale: {pred} (Confiance: {conf}%)")
                    
                    return Prediction(
                        region=match.region,
                        competition=match.competition,
                        match=f"{match.home_team} vs {match.away_team}",
                        time=match.commence_time.astimezone(timezone(timedelta(hours=1))).strftime("%H:%M"),
                        predicted_score1=match.predicted_score1,
                        predicted_score2=match.predicted_score2,
                        prediction=pred,
                        confidence=conf,
                        home_odds=match.home_odds,
                        draw_odds=match.draw_odds,
                        away_odds=match.away_odds
                    )

            print("❌ Pas de prédiction fiable")
            return None

        except Exception as e:
            print(f"❌ Erreur lors de l'analyse avec Claude: {str(e)}")
            return None

    def analyze_match_fallback(self, match: Match, stats: str) -> Optional[Prediction]:
        """Version moins stricte de l'analyse pour garantir des prédictions minimales"""
        print(f"\n4️⃣b ANALYSE FALLBACK POUR {match.home_team} vs {match.away_team}")
        
        try:
            # Utiliser un prompt moins strict et une température plus élevée
            prompt = f"""ANALYSE SIMPLIFIÉE: {match.home_team} vs {match.away_team}
COMPÉTITION: {match.competition}
SCORES EXACTS PRÉDITS: {match.predicted_score1} et {match.predicted_score2}
COTES: Victoire {match.home_team}: {match.home_odds}, Match nul: {match.draw_odds}, Victoire {match.away_team}: {match.away_odds}

DONNÉES STATISTIQUES:
{stats}

CONSIGNES:
1. Choisir la prédiction la plus probable parmi: {', '.join(self.available_predictions)}
2. Utiliser une confiance minimale de 70%
3. Privilégier les prédictions de buts (+1.5, +2.5) ou doubles chances (1X, X2) si les victoires directes sont incertaines

FORMAT REQUIS:
PREDICTION: [une option de la liste]
CONFIANCE: [pourcentage]"""

            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                temperature=0.7,  # Température plus élevée pour plus de créativité
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text
            prediction = re.search(r"PREDICTION:\s*(.*)", response)
            confidence = re.search(r"CONFIANCE:\s*(\d+)", response)

            if all([prediction, confidence]):
                pred = prediction.group(1).strip()
                conf = min(100, max(70, int(confidence.group(1))))  # Minimum 70% au lieu de 80%
                
                if any(p.lower() in pred.lower() for p in self.available_predictions):
                    # Trouver la prédiction exacte dans la liste
                    for available_pred in self.available_predictions:
                        if available_pred.lower() in pred.lower():
                            pred = available_pred
                            break
                    
                    # Vérifications moins strictes pour le fallback
                    if pred == "X":
                        print("⚠️ Prédiction de match nul convertie en double chance 1X")
                        pred = "1X"  # Convertir match nul en 1X au lieu de rejeter
                    
                    print(f"✅ Prédiction fallback: {pred} (Confiance: {conf}%)")
                    
                    return Prediction(
                        region=match.region,
                        competition=match.competition,
                        match=f"{match.home_team} vs {match.away_team}",
                        time=match.commence_time.astimezone(timezone(timedelta(hours=1))).strftime("%H:%M"),
                        predicted_score1=match.predicted_score1,
                        predicted_score2=match.predicted_score2,
                        prediction=pred,
                        confidence=conf,
                        home_odds=match.home_odds,
                        draw_odds=match.draw_odds,
                        away_odds=match.away_odds
                    )

            print("❌ Pas de prédiction fallback fiable")
            return None

        except Exception as e:
            print(f"❌ Erreur lors de l'analyse fallback: {str(e)}")
            return None

    def _format_predictions_message(self, predictions: List[Prediction]) -> str:
        # Date du jour formatée
        current_date = datetime.now().strftime('%d/%m/%Y')
        
        # En-tête du message avec formatage en gras
        msg = f"*🤖 AL VE AI BOT - PRÉDICTIONS DU {current_date} 🤖*\n\n"

        for i, pred in enumerate(predictions, 1):
            # Formatage des éléments avec gras et italique - SANS LES COTES
            msg += (
                f"*📊 MATCH #{i}*\n"
                f"🏆 _{pred.competition}_\n"
                f"*⚔️ {pred.match}*\n"
                f"⏰ Coup d'envoi : _{pred.time}_\n"
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

        print("\n5️⃣ ENVOI DES PRÉDICTIONS")
        
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
            all_matches = self.fetch_matches()
            if not all_matches:
                print("❌ Aucun match trouvé pour aujourd'hui")
                return

            predictions = []
            fallback_candidates = []  # Pour stocker les matchs avec leurs analyses pour un usage en secours
            
            # Sélectionner les TOP matchs pour analyse (au moins MIN_PREDICTIONS)
            selected_matches = all_matches[:max(self.config.MIN_PREDICTIONS, self.config.MAX_MATCHES)]
            print(f"\n📋 {len(selected_matches)} matchs prioritaires sélectionnés pour analyse complète")
            
            # Traiter chaque match sélectionné
            for match in selected_matches:
                print(f"\n⚽ TRAITEMENT DE {match.home_team} vs {match.away_team}")
                
                # Obtenir les deux scores exacts probables
                scores = self.get_predicted_scores(match)
                if not scores:
                    print(f"⚠️ Impossible d'obtenir des scores exacts. Match ignoré.")
                    continue
                    
                match.predicted_score1, match.predicted_score2 = scores
                
                # Obtenir les statistiques
                stats = self.get_match_stats(match)
                if not stats:
                    print(f"⚠️ Impossible d'obtenir des statistiques. Match ignoré.")
                    continue
                
                # Stocker ce match et ses données pour un usage potentiel en fallback
                fallback_candidates.append((match, stats))
                
                # Analyser le match et obtenir une prédiction
                prediction = self.analyze_match(match, stats)
                if prediction:
                    predictions.append(prediction)
                    print(f"✅ Prédiction {len(predictions)}/{self.config.MIN_PREDICTIONS} obtenue")
                
                # Attendre un peu entre chaque analyse pour ne pas surcharger les API
                await asyncio.sleep(5)  # Attendre 5 secondes entre chaque match
            
            # Seconde passe - utiliser l'approche fallback si nécessaire
            if len(predictions) < self.config.MIN_PREDICTIONS and fallback_candidates:
                print(f"\n⚠️ Seulement {len(predictions)} prédictions obtenues. Tentative avec critères assouplis...")
                
                # Trier les candidats fallback par priorité (qui ne sont pas déjà dans les prédictions)
                remaining_candidates = [
                    (match, stats) for match, stats in fallback_candidates 
                    if not any(p.match == f"{match.home_team} vs {match.away_team}" for p in predictions)
                ]
                remaining_candidates.sort(key=lambda x: -x[0].priority)
                
                for match, stats in remaining_candidates:
                    if len(predictions) >= self.config.MIN_PREDICTIONS:
                        break
                        
                    print(f"\n⚽ SECONDE TENTATIVE POUR {match.home_team} vs {match.away_team}")
                    
                    # Analyser avec des critères plus souples
                    prediction = self.analyze_match_fallback(match, stats)
                    if prediction:
                        predictions.append(prediction)
                        print(f"✅ Prédiction fallback {len(predictions)}/{self.config.MIN_PREDICTIONS} obtenue")
                    
                    await asyncio.sleep(3)
            
            print(f"\n📊 {len(selected_matches)} matchs traités, {len(predictions)} prédictions obtenues")
            
            # Si toujours pas assez de prédictions et qu'il reste des matchs non traités
            if len(predictions) < self.config.MIN_PREDICTIONS and len(all_matches) > len(selected_matches):
                print(f"\n⚠️ Tentative avec des matchs supplémentaires...")
                
                # Prendre des matchs supplémentaires qui n'ont pas encore été analysés
                additional_matches = all_matches[len(selected_matches):len(selected_matches) + (self.config.MIN_PREDICTIONS - len(predictions))]
                
                for match in additional_matches:
                    if len(predictions) >= self.config.MIN_PREDICTIONS:
                        break
                        
                    print(f"\n⚽ TRAITEMENT SUPPLÉMENTAIRE DE {match.home_team} vs {match.away_team}")
                    
                    # Obtenir les deux scores exacts probables
                    scores = self.get_predicted_scores(match)
                    if not scores:
                        continue
                        
                    match.predicted_score1, match.predicted_score2 = scores
                    
                    # Obtenir les statistiques
                    stats = self.get_match_stats(match)
                    if not stats:
                        continue
                    
                    # Tenter d'abord l'analyse standard
                    prediction = self.analyze_match(match, stats)
                    if not prediction:
                        # Si échec, tenter l'analyse fallback
                        prediction = self.analyze_match_fallback(match, stats)
                    
                    if prediction:
                        predictions.append(prediction)
                        print(f"✅ Prédiction supplémentaire {len(predictions)}/{self.config.MIN_PREDICTIONS} obtenue")
                    
                    await asyncio.sleep(5)
            
            # Message final sur le nombre de prédictions
            if len(predictions) < self.config.MIN_PREDICTIONS:
                print(f"\n⚠️ Impossible d'atteindre le minimum de {self.config.MIN_PREDICTIONS} prédictions. Envoi de {len(predictions)} prédictions.")
            else:
                print(f"\n✅ {len(predictions)}/{self.config.MIN_PREDICTIONS} prédictions obtenues avec succès.")
            
            if predictions:
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
