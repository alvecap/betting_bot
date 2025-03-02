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
import pytz
import os  # Pour les variables d'environnement

# Configuration de base
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                   handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("betting_bot.log")])
logger = logging.getLogger(__name__)

@dataclass
class Config:
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str
    ODDS_API_KEY: str
    PERPLEXITY_API_KEY: str
    CLAUDE_API_KEY: str
    MAX_MATCHES: int = 5
    TIMEZONE: str = "Africa/Brazzaville"  # Fuseau horaire par défaut (UTC+1)
    EXECUTION_HOUR: int = 18
    EXECUTION_MINUTE: int = 35

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
    predicted_score: str = ""

@dataclass
class Prediction:
    region: str
    competition: str
    match: str
    time: str
    predicted_score: str
    prediction: str
    confidence: int

class BettingBot:
    def __init__(self, config: Config):
        self.config = config
        self.bot = telegram.Bot(token=config.TELEGRAM_BOT_TOKEN)
        self.claude_client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        self.timezone = pytz.timezone(config.TIMEZONE)
        self.available_predictions = [
            "1X", "X2", "12", 
            "+1.5 buts", "+2.5 buts", "-3.5 buts",
            "Les deux équipes marquent", 
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
        logger.info(f"Bot initialisé avec fuseau horaire: {config.TIMEZONE}")

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
    def fetch_matches(self) -> List[Match]:
        logger.info("1️⃣ RÉCUPÉRATION DES MATCHS...")
        url = "https://api.the-odds-api.com/v4/sports/soccer/odds/"
        params = {
            "apiKey": self.config.ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            matches_data = response.json()
            logger.info(f"✅ {len(matches_data)} matchs récupérés")

            # Utiliser UTC pour comparer avec les données de l'API
            current_time = datetime.now(timezone.utc)
            matches = []

            for match_data in matches_data:
                commence_time = datetime.fromisoformat(match_data["commence_time"].replace('Z', '+00:00'))
                # Prendre les matchs des prochaines 24 heures
                if 0 < (commence_time - current_time).total_seconds() <= 86400:
                    competition = self._get_league_name(match_data.get("sport_title", "Unknown"))
                    matches.append(Match(
                        home_team=match_data["home_team"],
                        away_team=match_data["away_team"],
                        competition=competition,
                        region=competition.split()[-1] if " " in competition else competition,
                        commence_time=commence_time,
                        bookmakers=match_data.get("bookmakers", []),
                        all_odds=match_data.get("bookmakers", []),
                        priority=self.top_leagues.get(competition, 0)
                    ))

            if not matches:
                logger.warning("Aucun match trouvé pour les prochaines 24 heures")
                return []

            # Trier les matchs par priorité et heure de début
            matches.sort(key=lambda x: (-x.priority, x.commence_time))
            
            # Prendre les meilleurs matchs
            top_matches = matches[:self.config.MAX_MATCHES]
            
            logger.info(f"✅ {len(top_matches)} meilleurs matchs sélectionnés")
            for match in top_matches:
                logger.info(f"- {match.home_team} vs {match.away_team} ({match.competition})")
                
            return top_matches

        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des matchs: {str(e)}", exc_info=True)
            return []

    def get_match_stats(self, match: Match) -> Optional[str]:
        logger.info(f"2️⃣ ANALYSE DE {match.home_team} vs {match.away_team}")
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [{
                        "role": "user", 
                        "content": f"""Analyse détaillée pour {match.home_team} vs {match.away_team} ({match.competition}):

1. FORME:
- 5 derniers matchs de chaque équipe
- Buts marqués/encaissés par match
- Résultats domicile/extérieur

2. CONFRONTATIONS DIRECTES:
- Historique des 5 dernières rencontres
- Tendances des scores
- Statistiques de buts dans ces matchs

3. STATISTIQUES IMPORTANTES:
- Moyenne de buts par match
- % matchs avec +1.5 buts
- % matchs avec +2.5 buts
- % matchs avec -3.5 buts
- % victoires/nuls/défaites
- Performance à domicile/extérieur

4. EFFECTIF:
- Blessés et suspendus
- Joueurs clés disponibles

5. CONTEXTE DU MATCH:
- Enjeu sportif
- Position au classement
- Série en cours"""
                    }],
                    "max_tokens": 800,
                    "temperature": 0.2
                },
                timeout=20
            )
            response.raise_for_status()
            stats = response.json()["choices"][0]["message"]["content"]
            logger.info("✅ Statistiques récupérées")
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des statistiques: {str(e)}", exc_info=True)
            return None

    def get_predicted_score(self, match: Match) -> str:
        logger.info(f"3️⃣ OBTENTION DU SCORE EXACT PROBABLE POUR {match.home_team} vs {match.away_team}")
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [{
                        "role": "user", 
                        "content": f"""Quel est le score exact le plus probable pour le match {match.home_team} vs {match.away_team} qui aura lieu le {match.commence_time.strftime('%d/%m/%Y')} en {match.competition}? 

Recherche les prédictions d'experts et les sites spécialisés. Réponds uniquement au format "X-Y" où X est le nombre de buts de l'équipe à domicile et Y est le nombre de buts de l'équipe à l'extérieur. Ne donne aucune autre information."""
                    }],
                    "max_tokens": 50,
                    "temperature": 0.1
                },
                timeout=15
            )
            response.raise_for_status()
            predicted_score = response.json()["choices"][0]["message"]["content"].strip()
            
            # Vérifier que le format est correct (X-Y)
            if re.match(r'^\d+-\d+$', predicted_score):
                logger.info(f"✅ Score probable obtenu: {predicted_score}")
                return predicted_score
            else:
                # Tenter d'extraire un format de score s'il est inclus dans une phrase
                score_match = re.search(r'(\d+)\s*-\s*(\d+)', predicted_score)
                if score_match:
                    clean_score = f"{score_match.group(1)}-{score_match.group(2)}"
                    logger.info(f"✅ Score probable extrait: {clean_score}")
                    return clean_score
                else:
                    logger.warning("❌ Format de score invalide, utilisation d'un score par défaut")
                    return "1-1"  # Score par défaut
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'obtention du score probable: {str(e)}", exc_info=True)
            return "1-1"  # Score par défaut en cas d'erreur

    def analyze_match(self, match: Match, stats: str) -> Optional[Prediction]:
        logger.info(f"4️⃣ ANALYSE AVEC CLAUDE POUR {match.home_team} vs {match.away_team}")
        
        try:
            prompt = f"""ANALYSE APPROFONDIE: {match.home_team} vs {match.away_team}
COMPÉTITION: {match.competition}
SCORE EXACT PRÉDIT: {match.predicted_score}

DONNÉES STATISTIQUES:
{stats}

CONSIGNES:
1. Analyser en profondeur les statistiques fournies et le score exact prédit
2. Évaluer les tendances et performances des équipes
3. Considérer le score exact prédit par les experts
4. Choisir la prédiction LA PLUS SÛRE parmi: {', '.join(self.available_predictions)}
5. Justifier avec précision
6. Confiance minimale de 80%

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
                            
                    logger.info(f"✅ Prédiction: {pred} (Confiance: {conf}%)")
                    
                    # Convertir l'heure dans le fuseau horaire local
                    local_time = match.commence_time.astimezone(self.timezone)
                    
                    return Prediction(
                        region=match.region,
                        competition=match.competition,
                        match=f"{match.home_team} vs {match.away_team}",
                        time=local_time.strftime("%H:%M"),
                        predicted_score=match.predicted_score,
                        prediction=pred,
                        confidence=conf
                    )

            logger.warning("❌ Pas de prédiction fiable")
            return None

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse avec Claude: {str(e)}", exc_info=True)
            return None

    def _format_predictions_message(self, predictions: List[Prediction]) -> str:
        # Date du jour formatée
        current_date = datetime.now(self.timezone).strftime('%d/%m/%Y')
        
        # En-tête du message avec formatage en gras
        msg = f"*🤖 AL VE AI BOT - PRÉDICTIONS DU {current_date} 🤖*\n\n"

        for i, pred in enumerate(predictions, 1):
            # Formatage des éléments avec gras et italique
            msg += (
                f"*📊 MATCH #{i}*\n"
                f"🏆 _{pred.competition}_\n"
                f"*⚔️ {pred.match}*\n"
                f"⏰ Coup d'envoi : _{pred.time}_\n"
                f"🔮 Score prédit : *{pred.predicted_score}*\n"
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
            logger.warning("❌ Aucune prédiction à envoyer")
            return

        logger.info("5️⃣ ENVOI DES PRÉDICTIONS")
        
        try:
            message = self._format_predictions_message(predictions)
            
            # Envoyer un message avec formatage Markdown
            await self.bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode="Markdown",  # Activer le formatage Markdown
                disable_web_page_preview=True
            )
            logger.info("✅ Prédictions envoyées!")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'envoi des prédictions: {str(e)}", exc_info=True)

    async def run(self) -> None:
        try:
            current_time = datetime.now(self.timezone)
            logger.info(f"\n=== 🤖 AL VE AI BOT - GÉNÉRATION DES PRÉDICTIONS ({current_time.strftime('%H:%M')}) ===")
            
            # Envoyer un message test pour vérifier que le bot fonctionne
            try:
                await self.bot.send_message(
                    chat_id=self.config.TELEGRAM_CHAT_ID,
                    text=f"🔄 Début de l'exécution du bot à {current_time.strftime('%H:%M')} (fuseau horaire: {self.config.TIMEZONE})"
                )
                logger.info("✅ Message de test envoyé")
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'envoi du message de test: {str(e)}", exc_info=True)
            
            matches = self.fetch_matches()
            if not matches:
                logger.warning("❌ Aucun match trouvé pour aujourd'hui")
                return

            predictions = []
            for match in matches:
                # Obtenir le score exact probable
                match.predicted_score = self.get_predicted_score(match)
                
                # Obtenir les statistiques
                stats = self.get_match_stats(match)
                if stats:
                    prediction = self.analyze_match(match, stats)
                    if prediction:
                        predictions.append(prediction)
                        
                # Attendre un peu entre chaque analyse pour ne pas surcharger les API
                await asyncio.sleep(2)

            if predictions:
                # Envoyer les prédictions une seule fois
                await self.send_predictions(predictions)
                logger.info("=== ✅ EXÉCUTION TERMINÉE ===")
            else:
                logger.warning("❌ Aucune prédiction fiable n'a pu être générée")
                # Envoyer un message d'erreur
                await self.bot.send_message(
                    chat_id=self.config.TELEGRAM_CHAT_ID,
                    text="❌ Aucune prédiction fiable n'a pu être générée pour aujourd'hui."
                )

        except Exception as e:
            logger.error(f"❌ ERREUR GÉNÉRALE: {str(e)}", exc_info=True)
            # Tenter d'envoyer un message d'erreur
            try:
                await self.bot.send_message(
                    chat_id=self.config.TELEGRAM_CHAT_ID,
                    text=f"❌ Une erreur s'est produite lors de l'exécution du bot: {str(e)}"
                )
            except:
                pass

async def scheduler():
    # Charger la configuration depuis les variables d'environnement
    config = Config(
        TELEGRAM_BOT_TOKEN=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        TELEGRAM_CHAT_ID=os.environ.get("TELEGRAM_CHAT_ID", ""),
        ODDS_API_KEY=os.environ.get("ODDS_API_KEY", ""),
        PERPLEXITY_API_KEY=os.environ.get("PERPLEXITY_API_KEY", ""),
        CLAUDE_API_KEY=os.environ.get("CLAUDE_API_KEY", ""),
        MAX_MATCHES=int(os.environ.get("MAX_MATCHES", "5")),
        TIMEZONE=os.environ.get("TIMEZONE", "Africa/Brazzaville"),
        EXECUTION_HOUR=int(os.environ.get("EXECUTION_HOUR", "09")),
        EXECUTION_MINUTE=int(os.environ.get("EXECUTION_MINUTE", "35"))
    )
    
    # Valider les clés API
    if not all([config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, config.ODDS_API_KEY, 
                config.PERPLEXITY_API_KEY, config.CLAUDE_API_KEY]):
        logger.error("❌ Configuration incomplète: certaines clés API sont manquantes")
        return
    
    timezone = pytz.timezone(config.TIMEZONE)
    logger.info(f"Scheduler démarré. Exécution prévue à {config.EXECUTION_HOUR}:{config.EXECUTION_MINUTE:02d} ({config.TIMEZONE})")
    
    # Exécuter le bot immédiatement au démarrage pour tester
    logger.info("Exécution initiale de test...")
    bot = BettingBot(config)
    await bot.run()
    
    while True:
        # Heure actuelle dans le fuseau horaire spécifié
        now = datetime.now(timezone)
        
        # Calculer l'heure dans le fuseau horaire UTC
        now_utc = datetime.now(pytz.UTC)
        
        # Log détaillé des heures
        if now.minute % 10 == 0:  # Log toutes les 10 minutes
            logger.info(f"Heure actuelle: {now.strftime('%Y-%m-%d %H:%M:%S')} ({config.TIMEZONE})")
            logger.info(f"Heure UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
            logger.info(f"Prochaine exécution à: {config.EXECUTION_HOUR}:{config.EXECUTION_MINUTE:02d} ({config.TIMEZONE})")

        # Vérifier si c'est l'heure d'exécution
        if now.hour == config.EXECUTION_HOUR and now.minute == config.EXECUTION_MINUTE:
            logger.info(f"⏰ Déclenchement du bot à {now.strftime('%Y-%m-%d %H:%M:%S')} ({config.TIMEZONE})")
            bot = BettingBot(config)
            await bot.run()

            # Attendre 1 minute pour éviter les exécutions multiples
            await asyncio.sleep(60)
        
        # Attendre 30 secondes avant de vérifier à nouveau
        await asyncio.sleep(30)

async def main():
    # Mode test: exécution immédiate une seule fois
    if os.environ.get("TEST_MODE") == "1":
        logger.info("Mode test activé: exécution immédiate")
        config = Config(
            TELEGRAM_BOT_TOKEN=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            TELEGRAM_CHAT_ID=os.environ.get("TELEGRAM_CHAT_ID", ""),
            ODDS_API_KEY=os.environ.get("ODDS_API_KEY", ""),
            PERPLEXITY_API_KEY=os.environ.get("PERPLEXITY_API_KEY", ""),
            CLAUDE_API_KEY=os.environ.get("CLAUDE_API_KEY", ""),
            MAX_MATCHES=int(os.environ.get("MAX_MATCHES", "5"))
        )
        bot = BettingBot(config)
        await bot.run()
    else:
        # Mode normal: scheduler
        await scheduler()

if __name__ == "__main__":
    asyncio.run(main())
